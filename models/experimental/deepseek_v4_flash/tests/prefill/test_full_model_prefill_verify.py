# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness check of the full prefill demo: the real 43-layer checkpoint on the Galaxy against a CPU reference.

It builds exactly the model of ``test_full_model_prefill_demo.py`` (8 x 4 Galaxy, ``DEEPSEEK_V4_PREFILL_STAGES``
pipeline stages of ``1 x 4`` TP4 submeshes, dense CSA, bf8 attention, bf4 routed experts) and runs the *same
kind of prompt* through it chunk by chunk, then compares with the fp32 HF-style reference of
``full_model_reference.py`` (the standalone V4 modules with the real weights, one layer resident at a time; it
is computed on the host beforehand and cached, so the device is not held while it runs).

Three views of the same run, all logged before anything is asserted:

1. *Teacher-forced, per layer.* Each layer gets the **reference's** input streams and its output is compared
   with the reference's output. This isolates the error of one layer (attention + MoE + hyper-connections) from
   the accumulated error of the layers before it, so a broken layer shows up at its own index.
2. *End-to-end, per layer.* The whole model runs on its own activations; after every layer the streams are
   compared with the reference's. This is what the demo actually computes; the error grows with depth
   (bf8/bf4 weights, bf16 activations), so the floors are looser.
3. *Logits.* PCC of the final logits over every token, the fraction of tokens whose argmax matches, whether the
   reference's top-1 is in the device's top 5, and the top-5 tokens after the last position from both.

Exactness: with ``dense_csa`` the device is the model's exact answer only while the compressed entries fit in
``index_topk`` (``T // 4 <= 512``, i.e. ``T <= 2048``), so the default prompt is 2048 tokens. For a longer
one the reference is switched to ``index_topk = inf`` (dense) so it is the same computation as the device, but
neither is then what the real model outputs.

Steps::

    # 1. the reference, on the host (~10-20 min, ~40 GB of RAM; no device); cached under the cache dir
    python -m models.experimental.deepseek_v4_flash.tests.prefill.full_model_reference --len 2048
    # 2. the device run (DEEPSEEK_V4_VERIFY_GENERATE=1 would compute a missing reference in-test, holding the card)
    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v4_flash/tests/prefill/test_full_model_prefill_verify.py

Knobs (environment): ``DEEPSEEK_V4_VERIFY_LEN`` (2048), ``DEEPSEEK_V4_VERIFY_CHUNK`` (1024),
``DEEPSEEK_V4_VERIFY_LAYERS`` (all; must match the reference's), ``DEEPSEEK_V4_PREFILL_STAGES`` (2),
``DEEPSEEK_V4_VERIFY_REF_DIR`` (reference cache dir), ``DEEPSEEK_V4_VERIFY_GENERATE`` (0), and the floors
``DEEPSEEK_V4_VERIFY_LAYER_PCC`` (0.95, teacher-forced, every layer), ``DEEPSEEK_V4_VERIFY_E2E_PCC`` (0.90,
end-to-end, every layer), ``DEEPSEEK_V4_VERIFY_LOGITS_PCC`` (0.85), ``DEEPSEEK_V4_VERIFY_TOP5`` (0.8, the
fraction of tokens whose reference top-1 is in the device top 5).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.tests.decode.test_full_model_decode_demo import (
    _CACHE_DIR,
    _DEFAULT_MODEL_DIR,
    _build_rope,
    _checkpoint_available,
)
from models.experimental.deepseek_v4_flash.tests.prefill.full_model_reference import (
    reference_config,
    run_reference,
    store_for,
)
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _DEFAULT_PROMPT_FILE,
    _TP_SIZE,
    _Progress,
    _build_prompt_ids,
    _env_int,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4PrefillModel
from models.experimental.deepseek_v4_flash.tt.prefill.weights import checkpoint_expert_provider, checkpoint_weights
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _token_pcc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of each row of ``x`` and ``y`` (``[T, N]``): a ``[T]`` fp64 tensor."""
    x = x.double() - x.double().mean(dim=1, keepdim=True)
    y = y.double() - y.double().mean(dim=1, keepdim=True)
    denom = (x.norm(dim=1) * y.norm(dim=1)).clamp_min(1e-30)
    return (x * y).sum(dim=1) / denom


@dataclass
class _Compare:
    """Running comparison of a device tensor with its reference over several chunks.

    ``pcc`` is over every element (from fp64 sums, exact however the chunks are cut); the per-token PCCs
    are kept to report the worst and the median token; ``rel`` is ``||dev - ref|| / ||ref||``.
    """

    n: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0
    tokens: list = field(default_factory=list)
    finite: bool = True

    def add(self, dev: torch.Tensor, ref: torch.Tensor) -> None:
        """``dev`` / ``ref``: ``[T, ...]`` for the same tokens."""
        dev = dev.reshape(dev.shape[0], -1)
        ref = ref.reshape(ref.shape[0], -1)
        self.finite = self.finite and bool(torch.isfinite(dev).all())
        dev = torch.nan_to_num(dev.double())
        ref = ref.double()
        self.n += dev.numel()
        self.sx += dev.sum().item()
        self.sy += ref.sum().item()
        self.sxx += (dev * dev).sum().item()
        self.syy += (ref * ref).sum().item()
        self.sxy += (dev * ref).sum().item()
        self.tokens.append(_token_pcc(dev, ref).float())

    @property
    def pcc(self) -> float:
        cov = self.sxy - self.sx * self.sy / self.n
        var_x = self.sxx - self.sx**2 / self.n
        var_y = self.syy - self.sy**2 / self.n
        return cov / math.sqrt(max(var_x * var_y, 1e-30))

    @property
    def rel(self) -> float:
        return math.sqrt(max(self.sxx - 2 * self.sxy + self.syy, 0.0) / max(self.syy, 1e-30))

    def per_token(self) -> torch.Tensor:
        return torch.cat(self.tokens)

    def summary(self) -> str:
        tok = self.per_token()
        return (
            f"pcc {self.pcc:.5f} | rel err {self.rel:.4f} | token pcc min {tok.min():.4f} "
            f"median {tok.median():.4f}{'' if self.finite else ' | NON-FINITE VALUES'}"
        )


def _upload_streams(host: torch.Tensor, dev) -> ttnn.Tensor:
    """``[1, T, hc, D]`` host streams -> bf16 TILE on ``dev``, replicated (what a layer takes)."""
    return ttnn.from_torch(
        host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev) if dev.get_num_devices() > 1 else None,
    )


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_full_model_prefill_verify(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    progress = _Progress(
        interval=float(os.environ.get("DEEPSEEK_V4_PREFILL_HEARTBEAT", 30)),
        stall=float(os.environ.get("DEEPSEEK_V4_PREFILL_STALL_SECS", 600)),
    )
    progress.verbose = False
    with progress:
        _run(mesh_device, progress, AutoTokenizer)


def _run(mesh_device, progress: _Progress, AutoTokenizer) -> None:
    target_len = _env_int("DEEPSEEK_V4_VERIFY_LEN", 2048)
    chunk_size = _env_int("DEEPSEEK_V4_VERIFY_CHUNK", 1024)
    for name, value in (("DEEPSEEK_V4_VERIFY_LEN", target_len), ("DEEPSEEK_V4_VERIFY_CHUNK", chunk_size)):
        if value <= 0 or value % ALIGNMENT:
            raise ValueError(f"{name}={value} must be a positive multiple of {ALIGNMENT}")
    layer_floor = _env_float("DEEPSEEK_V4_VERIFY_LAYER_PCC", 0.95)
    e2e_floor = _env_float("DEEPSEEK_V4_VERIFY_E2E_PCC", 0.90)
    logits_floor = _env_float("DEEPSEEK_V4_VERIFY_LOGITS_PCC", 0.85)
    top5_floor = _env_float("DEEPSEEK_V4_VERIFY_TOP5", 0.8)

    # --- checkpoint, prompt, reference ------------------------------------------------------------- #
    progress.step("[1/6] checkpoint, tokenizer, prompt")
    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = reference_config(loader, target_len)  # eager; index_topk widened when T needs it
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(_env_int("DEEPSEEK_V4_VERIFY_LAYERS", config.num_hidden_layers), config.num_hidden_layers)
    entry = json.loads(Path(os.environ.get("DEEPSEEK_V4_PREFILL_PROMPT", _DEFAULT_PROMPT_FILE)).read_text())[0]
    prompt_ids, info = _build_prompt_ids(tokenizer, entry, target_len)
    assert len(prompt_ids) % ALIGNMENT == 0
    ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    total = ids.shape[1]
    num_chunks = math.ceil(total / chunk_size)
    exact = total // 4 <= config.index_topk
    logger.info(
        f"verifying {num_layers} layers on a {total}-token prompt in {num_chunks} chunk(s) of {chunk_size} "
        f"({'exact CSA regime' if exact else 'dense CSA beyond index_topk: device and reference are dense, not the true model'})"
    )

    progress.step("[2/6] the CPU reference")
    store = store_for(ids, num_layers, config)
    if not store.complete():
        if os.environ.get("DEEPSEEK_V4_VERIFY_GENERATE") != "1":
            pytest.fail(
                f"no complete reference for this prompt in {store.dir}. Generate it on the host first (no device):\n"
                f"  python -m models.experimental.deepseek_v4_flash.tests.prefill.full_model_reference "
                f"--len {target_len}"
                + (f" --layers {num_layers}" if num_layers != config.num_hidden_layers else "")
                + "\nor set DEEPSEEK_V4_VERIFY_GENERATE=1 to compute it here (holds the card meanwhile)."
            )
        run_reference(loader, config, ids, store, progress=lambda m: progress(m, important=True))
    logger.info(f"reference: {store.dir}")

    # --- the mesh and the model: as in the demo -------------------------------------------------- #
    progress.step("[3/6] mesh, submeshes, the model")
    system_config = load_system_config(mesh_device=mesh_device).log()
    set_active_system_config(system_config)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    assert mesh_cols >= _TP_SIZE
    num_stages = _env_int("DEEPSEEK_V4_PREFILL_STAGES", 2)
    assert 1 <= num_stages <= mesh_rows
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0)) for i in range(num_stages)
    ]
    placement = plan_layer_placement(num_layers, num_stages, 1)
    layer_devices = [submeshes[k] for k in placement]
    rope = _build_rope(config, total)
    cache = WeightCache(os.path.join(_CACHE_DIR, os.path.basename(_DEFAULT_MODEL_DIR))) if _CACHE_DIR else None
    t0 = time.perf_counter()
    model = DeepSeekV4PrefillModel(
        config,
        checkpoint_weights(loader, config, num_layers),
        layer_devices[0],
        rope,
        expert_provider=checkpoint_expert_provider(loader),
        num_layers=num_layers,
        cache=cache,
        weight_dtype=_ATTENTION_WEIGHT_DTYPE,
        expert_dtype=system_config.decode.ttnn_weight_dtype,
        tp_size=_TP_SIZE,
        layer_devices=layer_devices,
        dense_csa=True,
        progress=progress,
    )
    model.synchronize("uploads")
    logger.info(f"model built in {time.perf_counter() - t0:.1f}s")

    kinds = [
        f"{config.layer_types[i].replace('_attention', '')}/{config.mlp_layer_types[i].replace('_moe', '')}"
        for i in range(num_layers)
    ]
    spans = [(s, min(total, s + chunk_size)) for s in range(0, total, chunk_size)]

    # --- 1. teacher-forced, per layer ---------------------------------------------------------------- #
    progress.step(f"[4/6] teacher-forced: every layer on the reference's own input ({num_layers} layers)")
    forced: list[_Compare] = []
    for i, layer in enumerate(model.layers):
        dev = model.layer_devices[i]
        ref_in, ref_out = store.layer_input(i), store.layer_output(i)
        state = layer.new_state()
        cmp = _Compare()
        for start, end in spans:
            x = _upload_streams(ref_in[:, start:end], dev)
            ids_dev = model._upload_ids(ids[:, start:end], dev)
            out = layer(x, state, ids_dev)
            cmp.add(model.to_host(out, dev)[0], ref_out[0, start:end])
            for t in (x, ids_dev, out):
                ttnn.deallocate(t)
        forced.append(cmp)
        logger.info(f"teacher-forced layer {i:2d} ({kinds[i]}): {cmp.summary()}")
        progress(f"teacher-forced layer {i + 1}/{num_layers} done", important=False)
        del ref_in, ref_out

    # --- 2. end to end, per layer and logits --------------------------------------------------------- #
    progress.step(f"[5/6] end-to-end: the whole model on its own activations, {num_chunks} chunk(s)")
    e2e = [_Compare() for _ in range(num_layers)]
    logit_cmp = _Compare()
    ref_layer_out = {}  # each layer's reference output, read on first use and kept for the later chunks (host RAM)
    states = model.new_state()
    ref_logits = store.logits()[0]  # [T, V]
    top1_match = top1_in_top5 = 0
    last_device_top = None

    for c, (start, end) in enumerate(spans):
        model._tag = f"verify chunk {c + 1}/{num_chunks}"

        def on_layer(i: int, out: ttnn.Tensor, dev) -> None:
            if i not in ref_layer_out:
                ref_layer_out[i] = store.layer_output(i)
            e2e[i].add(model.to_host(out, dev)[0], ref_layer_out[i][0, start:end])

        streams = model._stack(ids[:, start:end], states, on_layer=on_layer)
        logits = model.head(streams, last_only=False)
        ttnn.deallocate(streams)
        dev_logits = model.to_host(logits, model.head_device).reshape(end - start, -1)[:, : config.vocab_size]
        ttnn.deallocate(logits)
        ref_chunk = ref_logits[start:end]
        logit_cmp.add(dev_logits, ref_chunk)
        ref_top1 = ref_chunk.argmax(dim=-1)
        dev_top5 = dev_logits.topk(5, dim=-1).indices
        top1_match += int((dev_top5[:, 0] == ref_top1).sum())
        top1_in_top5 += int((dev_top5 == ref_top1[:, None]).any(dim=-1).sum())
        if end == total:
            last_device_top = dev_logits[-1].topk(5)
        logger.info(f"end-to-end chunk {c + 1}/{num_chunks} [{start}, {end}) compared")

    # --- report -------------------------------------------------------------------------------------- #
    progress.step("[6/6] report")
    lines = [
        "",
        f"{'layer':>5} {'kind':<26} {'teacher-forced pcc':>19} {'min tok':>8} {'end-to-end pcc':>15} {'rel err':>8}",
    ]
    for i in range(num_layers):
        lines.append(
            f"{i:>5} {kinds[i]:<26} {forced[i].pcc:>19.5f} {forced[i].per_token().min():>8.4f} "
            f"{e2e[i].pcc:>15.5f} {e2e[i].rel:>8.4f}"
        )
    logger.info("\n".join(lines))

    top5_rate = top1_in_top5 / total
    ref_last = ref_logits[-1].topk(5)
    fmt = lambda top: ", ".join(  # noqa: E731
        f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices)
    )
    logger.info(
        "\n".join(
            [
                "",
                "=== logits ===",
                f"logits            : {logit_cmp.summary()}",
                f"argmax match      : {top1_match}/{total} tokens ({top1_match / total:.1%})",
                f"ref top-1 in top-5: {top1_in_top5}/{total} tokens ({top5_rate:.1%})",
                f"last token, reference top 5: {fmt(ref_last)}",
                f"last token, device    top 5: {fmt(last_device_top)}",
            ]
        )
    )

    failures = []
    for i in range(num_layers):
        if not forced[i].finite or forced[i].pcc < layer_floor:
            failures.append(f"teacher-forced layer {i} ({kinds[i]}): pcc {forced[i].pcc:.5f} < {layer_floor}")
        if not e2e[i].finite or e2e[i].pcc < e2e_floor:
            failures.append(f"end-to-end layer {i} ({kinds[i]}): pcc {e2e[i].pcc:.5f} < {e2e_floor}")
    if not logit_cmp.finite or logit_cmp.pcc < logits_floor:
        failures.append(f"logits pcc {logit_cmp.pcc:.5f} < {logits_floor}")
    if top5_rate < top5_floor:
        failures.append(f"reference top-1 in device top-5 for {top5_rate:.1%} of tokens < {top5_floor:.0%}")
    progress.step("done")
    assert not failures, "prefill verification failed:\n  " + "\n  ".join(failures)
