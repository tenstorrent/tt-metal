# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced-decode A/B over the **reduced** full-model variant, one arm per build.

Every candidate this stage considers for the decode step is measured here, on the
same two real layers (one sliding, one full attention) plus the *real* terminal
path -- real embedding table and gather, real terminal norm, real BFP4 LM head at
the real padded vocab, real softcap, real sampler, real traces.  The reduced build
is what makes an A/B affordable: a 52-layer build is ~160 s of host weight packing
per arm, two layers is ~10 s, and the candidates all change per-layer or terminal
work, which composes.

Arms:

``base``                the shipped full-model configuration
``no_softcap_l1``       tanh + scalar multiply on DRAM-interleaved logits (the
                        full-model stage's order), against the shipped L1 form
``no_embed_sharded``    decode embedding gather to DRAM + ``interleaved_to_sharded``
``mlpN``                MLP gate/up output shard grid of N cores, with ``mlp_down``'s
                        ``in0_block_w`` moved to the largest legal divisor of
                        ``5120 / 32 / N``.  This is the SwiGLU multiply's core count:
                        the multiply carries an SFPU SiLU and costs 18.0 us on the
                        16-core grid against 1.9 us for a plain 6656-wide add on the
                        same grid.

Correctness: every arm's decode logits are compared against ``base`` on the same
seeded prompt (top-1 agreement over the whole vocab plus PCC), so a faster arm that
moved the model is visible rather than silent.

Usage::

    python doc/optimized_full_model/bench/decode_ab.py --arms base,no_softcap_l1,mlp32
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt import model as model_mod  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt import optimized_decoder as dec_mod  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    MULTICHIP_DECODE_MATMUL,
    close_multichip_mesh,
    open_multichip_mesh,
)

OUT = ROOT / "doc/optimized_full_model"
INTERMEDIATE_TILES = 5120 // 32  # 160; the MLP gate/up output width in tiles


def say(*args) -> None:
    print(*args, flush=True)


def mlp_matmul_table(cores: int) -> dict:
    """``MULTICHIP_DECODE_MATMUL`` with the MLP gate/up output grid moved to ``cores``.

    ``mlp_down`` consumes that output, so its ``in0_block_w`` must divide the new
    per-core K-tile count ``160 / cores``; the largest legal value is taken, which
    is what the shipped 16-core entry does (``160 / 16 = 10``).
    """
    if INTERMEDIATE_TILES % cores:
        raise ValueError(f"{cores} cores does not divide {INTERMEDIATE_TILES} intermediate tiles")
    down_block = INTERMEDIATE_TILES // cores
    table = dict(MULTICHIP_DECODE_MATMUL)
    for dtype in {k[1] for k in table}:
        for role in ("mlp_gate", "mlp_up"):
            _, in0 = table[(role, dtype)]
            table[(role, dtype)] = (cores, in0)
        _, _ = table[("mlp_down", dtype)]
        table[("mlp_down", dtype)] = (cores, down_block)
    return table


def arm_kwargs(arm: str) -> tuple[dict, dict]:
    """``(build_kwargs, module_flags)`` for one arm."""
    if arm == "base":
        return {}, {}
    if arm == "no_softcap_l1":
        return {}, {"LM_HEAD_SOFTCAP_IN_L1": False}
    if arm == "no_embed_sharded":
        return {}, {"EMBED_DECODE_GATHER_SHARDED": False}
    if arm == "full_model_stage":
        # Every decode-path change this stage ships, reverted together: the
        # full-model stage's shipped configuration.
        return {}, {
            "LM_HEAD_SOFTCAP_IN_L1": False,
            "EMBED_DECODE_GATHER_SHARDED": False,
            "DECODE_SWIGLU_MUL_CORES": None,
        }
    if arm == "terminal_only":
        return {}, {"DECODE_SWIGLU_MUL_CORES": None}
    if arm.startswith("swiglu"):
        return {}, {"DECODE_SWIGLU_MUL_CORES": int(arm[6:])}
    if arm.startswith("mlp"):
        return {"decoder_kwargs": {"decode_matmul": mlp_matmul_table(int(arm[3:]))}}, {}
    raise SystemExit(f"unknown arm {arm!r}")


def measure(generator, prompt, *, rounds: int, replays: int) -> dict:
    mesh = generator.mesh_device
    generator.reset()
    generator.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
    ttnn.synchronize_device(mesh)

    logits_only = []
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        started = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(mesh, generator._trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        logits_only.append((time.perf_counter() - started) / replays * 1e3)

    slot = next(iter(generator.sampling._trace_states.values()))
    sampling = []
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        started = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(mesh, slot["id"], cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        sampling.append((time.perf_counter() - started) / replays * 1e3)

    token_out = []
    gen_len = 33
    for _ in range(rounds):
        generator.reset()
        started = time.perf_counter()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=1, enable_trace=True)
        one = time.perf_counter() - started
        generator.reset()
        started = time.perf_counter()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=gen_len, enable_trace=True)
        full = time.perf_counter() - started
        token_out.append((full - one) / (gen_len - 1) * 1e3)

    ttfts = []
    for _ in range(rounds):
        generator.reset()
        ttnn.synchronize_device(mesh)
        started = time.perf_counter()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=1, enable_trace=True)
        ttfts.append((time.perf_counter() - started) * 1e3)

    return {
        "traced_logits_only_ms": min(logits_only),
        "sampling_trace_ms": min(sampling),
        "token_out_ms": min(token_out),
        "ttft_ms": min(ttfts),
        "rounds": {"logits_only": logits_only, "sampling": sampling, "token_out": token_out, "ttft": ttfts},
    }


def decode_logits(generator, prompt) -> torch.Tensor:
    """Host logits for one decode step after prefilling ``prompt``, for correctness."""
    generator.reset()
    generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
    ttnn.execute_trace(generator.mesh_device, generator._trace_id, cq_id=0, blocking=True)
    gathered = generator.model.gather_and_untilize_logits(generator._trace_logits)
    host = generator.model.logits_to_torch(gathered, gathered=True)
    ttnn.deallocate(gathered)
    return host[0].to(torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="base,full_model_stage,mlp20,mlp32,mlp40")
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=32)
    parser.add_argument("--out", default="decode_ab.json")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    defaults = {
        model_mod: {n: getattr(model_mod, n) for n in ("LM_HEAD_SOFTCAP_IN_L1", "EMBED_DECODE_GATHER_SHARDED")},
        dec_mod: {n: getattr(dec_mod, n) for n in ("DECODE_SWIGLU_MUL_CORES",)},
    }

    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary: dict = {"arms": {}, "layers": args.layers, "prompt_len": args.prompt_len}
    reference: torch.Tensor | None = None
    try:
        torch.manual_seed(17)
        for arm in arms:
            build_kwargs, flags = arm_kwargs(arm)
            for module, names in defaults.items():
                for name, value in names.items():
                    setattr(module, name, flags.get(name, value))
            clear_generator_cache()
            generator = None
            try:
                started = time.perf_counter()
                generator = build_generator(
                    ROOT,
                    mesh,
                    max_seq_len=args.max_seq_len,
                    layer_indices=[int(i) for i in args.layers.split(",")],
                    reuse=False,
                    **build_kwargs,
                )
                build_s = time.perf_counter() - started
                torch.manual_seed(17)
                vocab = generator.model.config.vocab_size
                prompt = [int(t) for t in torch.randint(0, vocab, (args.prompt_len,)).tolist()]

                logits = decode_logits(generator, prompt)
                entry = measure(generator, prompt, rounds=args.rounds, replays=args.replays)
                entry["build_seconds"] = round(build_s, 1)
                if reference is None:
                    reference = logits
                    entry["vs_base"] = {"top1_same": True, "pcc": 1.0, "note": "reference arm"}
                else:
                    pcc = float(torch.corrcoef(torch.stack([logits, reference]))[0, 1])
                    entry["vs_base"] = {
                        "top1_same": bool(int(logits.argmax()) == int(reference.argmax())),
                        "base_top1": int(reference.argmax()),
                        "arm_top1": int(logits.argmax()),
                        "pcc": round(pcc, 9),
                        "max_abs_diff": round(float((logits - reference).abs().max()), 6),
                    }
                summary["arms"][arm] = entry
                say(
                    f"AB {arm:<18} logits_only={entry['traced_logits_only_ms']:.4f} "
                    f"sampling={entry['sampling_trace_ms']:.4f} token_out={entry['token_out_ms']:.4f} "
                    f"ttft={entry['ttft_ms']:.2f} pcc={entry['vs_base']['pcc']:.6f} "
                    f"top1_same={entry['vs_base']['top1_same']}"
                )
            except Exception as exc:  # noqa: BLE001
                summary["arms"][arm] = {"error": str(exc).splitlines()[-1][:400]}
                say(f"AB {arm:<18} FAILED {str(exc).splitlines()[-1][:200]}")
            finally:
                if generator is not None:
                    generator.teardown()
                    # Free the arm's weights before the next build: five 2-layer
                    # builds each carry the real 672 MB embedding and 190 MB head.
                    generator.model.deallocate()
                clear_generator_cache()
        say("AB_OK")
        return 0
    finally:
        for module, names in defaults.items():
            for name, value in names.items():
                setattr(module, name, value)
        path = OUT / args.out
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        say(f"AB summary -> {path}")
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
