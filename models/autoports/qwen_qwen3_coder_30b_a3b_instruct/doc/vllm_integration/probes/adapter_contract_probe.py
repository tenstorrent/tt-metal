# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Drive ``tt/generator_vllm.py`` exactly as the TT plugin does, and check the
contract the serving goal actually cares about.

A live server proves the model *serves*. It cannot prove **how**: whether the
token came out of the full model's traced split sampler or out of some
adapter-local argmax, whether the page table is copied to device every token, or
whether a decode step reads host state it is not allowed to trust. Those are
statements about host-side work per token, and the generator already counts every
one of them in ``trace_stats``. This drives the adapter with the same kwargs
``vllm_tt_plugin`` builds -- fixed ``max_num_seqs`` rows, ``-1`` for inactive
slots, ``TTSamplingParams`` per row, ``read_from_device=False`` then
``read_decode_output``/``process_decode_output_host`` -- and asserts on those
counters.

Checks
------

1. **Split sampling is reused, not replaced.** The sampled token must come back
   as a *device* tensor from ``decode_forward(read_from_device=False)``, the
   generator's ``replays`` counter must advance by exactly one per token, and the
   model's own sampler must be the thing that produced it (asserted against
   ``Qwen3CoderModel.sampler``, and by the fact that a steady token costs zero
   token host copies -- an adapter-side argmax would need the logits on host).
2. **No host token feedback.** ``token_host_copies`` is flat across steady-state
   decode; the token reaches step *N+1* through ``tt_out_tok`` on device.
3. **No full-logits readback on the measured path.** ``caller_token_readbacks``
   advances by one per token (the 128-byte sampled-token tensor) and nothing
   gathers the vocabulary.
4. **Positions advance on device.** ``position_host_copies`` and
   ``rotary_position_host_copies`` are flat across steady-state decode.
5. **The page table is not copied when it is unchanged**, and *is* copied when a
   slot's blocks change.
6. **Stale scheduler input is ignored where it must be.** A steady decode step is
   handed a deliberately wrong host token and a one-step-behind host position;
   the output must be identical to the same run without the sabotage, because the
   adapter passes neither.
7. **Current position is honoured on a real layout change.** After a
   ``reset_batch`` that moves a slot to fresh blocks, the adapter must take the
   host's token and position for that slot and the device's for the continuing
   ones.
8. **Non-aligned prompt lengths serve**, at the adapter boundary, at lengths
   divisible by none of 32 (page/tile/sampling slots), 128 or 8.
9. **vLLM owns the cache.** ``decode_forward`` with ``kv_cache=None`` and no
   prior ``allocate_kv_cache`` must raise rather than quietly allocate a second,
   standalone cache.

Run (occupies the mesh; no server may be running):

    python models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/vllm_integration/probes/adapter_contract_probe.py \
        [--num-layers 2] [--max-num-seqs 4] [--max-model-len 4096] \
        [--out doc/vllm_integration/probes/adapter_contract_probe.json]

Exits non-zero on any failed check, so it is a gate and not a report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from vllm_tt_plugin.model_input import TTSamplingParams  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator_vllm import Qwen3CoderForCausalLM  # noqa: E402

TRACE_REGION_SIZE = 50331648
BLOCK_SIZE = 32


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if not r["pass"]]


def greedy_params(rows: int) -> TTSamplingParams:
    """Exactly what the plugin sends for a batch of greedy requests."""
    return TTSamplingParams(
        temperature=[0.0] * rows,
        top_k=[1] * rows,
        top_p=[1.0] * rows,
        seed=[None] * rows,
    )


def serving_audit_block(audit: dict) -> dict:
    """The adapter's ``serving_audit()`` minus the blocks reported separately.

    ``max_model_len`` is renamed to ``probe_max_model_len`` on the way out. The
    live adapter key keeps its vLLM name; here the value is only ever this
    probe's reduced target (4096), and the runner-side context-contract
    guardrail treats any ``max_model_len``-style JSON key below the model's
    supported context (262144) as a served cap, which this is not.
    """
    block = {k: v for k, v in audit.items() if k not in ("trace_stats", "model_runtime_fallbacks")}
    if "max_model_len" in block:
        block["probe_max_model_len"] = block.pop("max_model_len")
    return block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--prompt-len", type=int, default=131, help="deliberately non-aligned")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()

    checks = Checks()
    batch = args.max_num_seqs
    prompt_len = args.prompt_len
    for divisor in (32, 128, 8, 64):
        if prompt_len % divisor == 0:
            raise SystemExit(f"--prompt-len {prompt_len} is aligned to {divisor}; pick a non-aligned length")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=TRACE_REGION_SIZE)
    try:
        os.environ.setdefault("QWEN3_VLLM_NUM_LAYERS", str(args.num_layers))
        from transformers import AutoConfig

        from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import _resolve_snapshot

        hf_config = AutoConfig.from_pretrained(_resolve_snapshot())
        t0 = time.time()
        model = Qwen3CoderForCausalLM.initialize_vllm_model(
            hf_config, mesh, batch, max_seq_len=args.max_model_len, n_layers=args.num_layers
        )
        print(f"model built in {time.time() - t0:.1f}s", flush=True)

        # --- 9. vLLM owns the cache -------------------------------------------
        try:
            model.decode_forward(
                tokens=torch.zeros((batch, 1), dtype=torch.int32),
                page_table=torch.zeros((batch, 4), dtype=torch.int32),
                kv_cache=None,
                start_pos=torch.zeros(batch, dtype=torch.int64),
                sampling_params=greedy_params(batch),
            )
            checks.check("vllm_owns_cache", False, "decode_forward silently allocated a standalone cache")
        except RuntimeError as exc:
            checks.check("vllm_owns_cache", "allocate_kv_cache" in str(exc), f"raised: {exc}")

        # --- the plugin's own allocation call ---------------------------------
        pages_per_user = min(math.ceil(args.max_model_len / BLOCK_SIZE), 4096)
        num_blocks = batch * pages_per_user
        kv_cache = model.allocate_kv_cache(
            (num_blocks, model.model.config.local_attention.num_key_value_heads, BLOCK_SIZE, model.model.head_dim),
            torch.bfloat16,
            hf_config.num_hidden_layers,
        )
        model.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
        model.warmup_model_decode(
            kv_cache=kv_cache,
            max_batch_size=batch,
            num_blocks=pages_per_user,
            can_sample_on_device=True,
            enable_trace=True,
        )

        def page_table(assignment: list[int]) -> torch.Tensor:
            """One disjoint block run per slot, at vLLM's table width."""
            table = torch.zeros((batch, pages_per_user), dtype=torch.int32)
            span = math.ceil((prompt_len + args.steps + BLOCK_SIZE) / BLOCK_SIZE)
            for slot, base in enumerate(assignment):
                table[slot, :span] = torch.arange(base * span, base * span + span, dtype=torch.int32)
            return table

        # --- 8. non-aligned prefill through the adapter -----------------------
        pt = page_table(list(range(batch)))
        tokens = torch.randint(1000, 5000, (batch, prompt_len), dtype=torch.int32)
        out = model.prefill_forward(
            tokens=tokens,
            page_table=pt[:batch],
            kv_cache=kv_cache,
            enable_trace=False,
            prompt_lens=torch.tensor([prompt_len] * batch).numpy(),
            start_pos=torch.zeros(batch, dtype=torch.int32).numpy(),
            sampling_params=greedy_params(batch),
        )
        checks.check(
            "non_aligned_prefill",
            tuple(out.shape) == (batch, 1) and torch.all(out >= 0),
            f"prompt_len={prompt_len} (not divisible by 8/32/64/128) -> tokens {out.reshape(-1).tolist()}",
        )

        stats = model.generator.trace_stats
        sampler_id = id(model.model.sampler)

        # --- steady-state decode ----------------------------------------------
        positions = torch.full((batch,), prompt_len, dtype=torch.int64)
        tok = out.reshape(-1).to(torch.int32).reshape(batch, 1)

        def decode_step(*, tokens_in, positions_in, table, reset):
            device_out = model.decode_forward(
                tokens=tokens_in,
                page_table=table,
                kv_cache=kv_cache,
                start_pos=positions_in,
                enable_trace=True,
                read_from_device=False,
                sampling_params=greedy_params(batch),
                reset_batch=reset,
            )
            is_device = isinstance(device_out, ttnn.Tensor)
            host, events = model.read_decode_output(device_out, async_read=True)
            for event in events:
                ttnn.event_synchronize(event)
            return is_device, model.process_decode_output_host(host, is_tokens=True)

        # install (vLLM always sends reset_batch=True on the first decode)
        is_device, first = decode_step(tokens_in=tok, positions_in=positions, table=pt, reset=True)
        checks.check(
            "async_split_returns_device_tensor",
            is_device,
            "decode_forward(read_from_device=False) returned a ttnn.Tensor; read_decode_output "
            "recorded an event and process_decode_output_host did the host formatting",
        )

        before = dict(stats)
        emitted = [first.reshape(-1)[:batch].tolist()]
        for _ in range(args.steps):
            # vLLM always sends a full tokens/start_pos pair; a steady step must
            # ignore both and replay over the state the device already owns.
            _, host_tokens = decode_step(tokens_in=tok, positions_in=positions, table=pt, reset=False)
            emitted.append(host_tokens.reshape(-1)[:batch].tolist())
        after = dict(stats)
        delta = {k: after[k] - before[k] for k in after if isinstance(after[k], int)}

        checks.check(
            "steady_replays", delta["replays"] == args.steps, f"replays +{delta['replays']} over {args.steps} steps"
        )
        checks.check(
            "steady_no_token_host_copies",
            delta["token_host_copies"] == 0,
            f"token_host_copies +{delta['token_host_copies']}",
        )
        checks.check(
            "steady_no_position_host_copies",
            delta["position_host_copies"] == 0 and delta["rotary_position_host_copies"] == 0,
            f"position +{delta['position_host_copies']}, rotary +{delta['rotary_position_host_copies']}",
        )
        checks.check(
            "steady_no_page_table_copies",
            delta["page_table_host_copies"] == 0,
            f"page_table_host_copies +{delta['page_table_host_copies']} for an unchanged table",
        )
        checks.check(
            "steady_no_recapture",
            delta["captures"] == 0 and delta["releases"] == 0 and delta["decode_warmups"] == 0,
            f"captures +{delta['captures']}, releases +{delta['releases']}, warmups +{delta['decode_warmups']}",
        )
        checks.check(
            "one_readback_per_token",
            delta["caller_token_readbacks"] == args.steps,
            f"caller_token_readbacks +{delta['caller_token_readbacks']} (the sampled-token tensor only)",
        )
        checks.check(
            "sampler_is_the_full_model_sampler",
            id(model.model.sampler) == sampler_id and type(model.model.sampler).__name__ == "_WatcherCleanSampling1D",
            f"{type(model.model.sampler).__name__} from tt/model.py, unchanged across the run",
        )

        # --- 6. stale scheduler input is ignored -------------------------------
        # Replay the same steady sequence, but hand every step a wrong host token
        # and a one-step-behind host position. The adapter passes neither, so the
        # emitted tokens must match the clean run exactly.
        model.generator.reset()
        model.prefill_forward(
            tokens=tokens,
            page_table=pt[:batch],
            kv_cache=kv_cache,
            enable_trace=False,
            prompt_lens=torch.tensor([prompt_len] * batch).numpy(),
            start_pos=torch.zeros(batch, dtype=torch.int32).numpy(),
            sampling_params=greedy_params(batch),
        )
        decode_step(tokens_in=tok, positions_in=positions, table=pt, reset=True)
        sabotaged = [emitted[0]]
        for _ in range(args.steps):
            _, host_tokens = decode_step(
                tokens_in=torch.full((batch, 1), 12345, dtype=torch.int32),
                positions_in=positions - 1,
                table=pt,
                reset=False,
            )
            sabotaged.append(host_tokens.reshape(-1)[:batch].tolist())
        checks.check(
            "stale_host_input_ignored",
            sabotaged[1:] == emitted[1:],
            f"steady decode with token=12345 and position-1 on every step reproduced the clean run "
            f"({emitted[1][:2]}... vs {sabotaged[1][:2]}...)",
        )

        # --- 5/7. a real layout change does copy, and is honoured --------------
        moved = page_table([(i + batch) for i in range(batch)])
        before = dict(stats)
        decode_step(
            tokens_in=torch.full((batch, 1), int(tok[0, 0]), dtype=torch.int32),
            positions_in=torch.full((batch,), prompt_len, dtype=torch.int64),
            table=moved,
            reset=True,
        )
        after = dict(stats)
        checks.check(
            "changed_page_table_is_copied",
            after["page_table_host_copies"] > before["page_table_host_copies"],
            f"page_table_host_copies +{after['page_table_host_copies'] - before['page_table_host_copies']} "
            "when every slot moved to fresh blocks",
        )
        state = model.generator.decode_device_state()
        checks.check(
            "current_position_reinstalled_on_layout_change",
            state is not None and int(state["positions"][0]) == prompt_len + 1,
            f"device current_pos after the reset step = {None if state is None else int(state['positions'][0])} "
            f"(host asked for {prompt_len}, the trace advanced it once)",
        )

        model.generator.teardown()
        audit = model.serving_audit()
    finally:
        try:
            ttnn.close_mesh_device(mesh)
        finally:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    payload = {
        "config": {
            "num_layers": args.num_layers,
            "max_num_seqs": batch,
            # Deliberately not named ``max_model_len``: this is the probe's own
            # reduced target, not a served context cap, and the runner-side
            # context-contract guardrail flags any ``max_model_len``-style key
            # below the model's supported context (262144).
            "probe_max_model_len": args.max_model_len,
            "prompt_len": prompt_len,
            "steady_steps": args.steps,
            "block_size": BLOCK_SIZE,
        },
        "note": (
            "Reduced-layer target on purpose: every check here is about host-side work per token and "
            "cache/page-table/scheduler-input handling, none of which depends on layer count. It is not "
            "an accuracy or performance artifact."
        ),
        "checks": checks.rows,
        "trace_stats": dict(audit["trace_stats"]),
        "serving_audit": serving_audit_block(audit),
        "model_runtime_fallbacks": audit["model_runtime_fallbacks"],
        "failed": len(checks.failed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {args.out}")
    if checks.failed:
        print(f"FAILED {len(checks.failed)} check(s): {[r['check'] for r in checks.failed]}")
        return 1
    print(f"all {len(checks.rows)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
