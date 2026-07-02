# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# HANG REPRO — Ideogram 4.0 ring-joint SDPA logical_n transition.
#
# The server hung on the FIRST genuinely-new prompt: several generations ran fine
# at one prompt length, then a different-length prompt wedged the board (worker
# thread asleep, device making no progress, no compilation). The denoiser runs
# under a CONSTANT padded sequence length L; only the runtime scalar `logical_n`
# (= num_img + n_text) that ring_joint_scaled_dot_product_attention uses for
# trailing-pad masking varies per prompt. The program_config depends only on the
# (constant) padded local seq len, so nothing recompiles — the suspicion is a
# logical_n-VALUE-dependent deadlock in the ring-joint SDPA.
#
# This test isolates exactly that: build the block ONCE at a fixed padded shape,
# warm the program cache, then run the SAME tensors through `forward` repeatedly
# with DIFFERENT `spatial_sequence_length` (logical_n) values. Everything but the
# scalar is constant. Each iteration logs before/after with a flush + device sync,
# so if the device wedges, the last "BEGIN"-without-"END" line names the exact
# logical_n, and the tt-metal watcher dump (run with TT_METAL_WATCHER=<sec>) shows
# which cores/kernels are stuck.
#
# Run under the safe-pytest + watcher harness (a hang won't wedge forever):
#   TT_METAL_WATCHER=5 timeout 900 \
#     pytest models/tt_dit/tests/models/ideogram4/test_hang_ideogram4.py -s -q \
#       -p no:cacheprovider --timeout=0
# Watcher dump lands in generated/watcher/watcher.log.
# =============================================================================

import os
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.transformer_ideogram4 import Ideogram4TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.padding import PaddingConfig
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard
from .test_transformer_ideogram4 import (
    ADALN_DIM,
    EMB_DIM,
    HEAD_DIM,
    INTERMEDIATE_SIZE,
    NORM_EPS,
    NUM_HEADS,
    _build_inputs,
    _sp_padded_len,
)

# 512px-equivalent image token count (fast to iterate); the hang is about the
# logical_n structure vs the ring/chunk layout, expected to be image_len-agnostic.
# Bump IMAGE_LEN to 16384 (2048px) via the parametrize id if 512px doesn't repro.
MAX_TEXT = 2048


def _log(msg):
    logger.info(msg)
    print(msg, flush=True)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links"),
    [pytest.param((4, 2), (4, 2), 0, 1, 1, id="sp4tp2")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("image_len", [1024, 16384], ids=["img1024_512px", "img16384_2048px"])
def test_ring_sdpa_logical_n_transition(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    image_len: int,
) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh.shape)[sp_axis]
    tp_factor = tuple(submesh.shape)[tp_axis]

    # Constant padded length L = _sp_padded_len(image_len + MAX_TEXT) — exactly what the
    # pipeline uses. Build inputs ONCE at full text_len=MAX_TEXT so tensors are max-size.
    seq_len = image_len + MAX_TEXT
    padded_len = _sp_padded_len(seq_len, sp_factor)
    x, adaln_input, _segment_ids, cos, sin, _bias = _build_inputs(1, MAX_TEXT, image_len, torch_dtype)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    ccl_manager = CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear)
    padding_config = (
        PaddingConfig.from_tensor_parallel_factor(NUM_HEADS, HEAD_DIM, tp_factor)
        if NUM_HEADS % tp_factor != 0
        else None
    )
    tt_block = Ideogram4TransformerBlock(
        hidden_size=EMB_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adaln_dim=ADALN_DIM,
        mesh_device=submesh,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    # Random weights: this is a liveness/hang test, not a correctness check.
    from ....reference.ideogram4 import modeling_ideogram4

    ref = modeling_ideogram4.Ideogram4TransformerBlock(
        hidden_size=EMB_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adanln_dim=ADALN_DIM,
    ).to(dtype=torch_dtype)
    tt_block.load_torch_state_dict(ref.state_dict())

    # Pad + shard the (constant) inputs once.
    x = torch.nn.functional.pad(x, (0, 0, 0, padded_len - seq_len))
    cos4 = torch.nn.functional.pad(cos.unsqueeze(1), (0, 0, 0, padded_len - seq_len))
    sin4 = torch.nn.functional.pad(sin.unsqueeze(1), (0, 0, 0, padded_len - seq_len))
    tt_x = bf16_tensor_2dshard(x, device=submesh, shard_mapping={sp_axis: 1, tp_axis: 2})
    tt_cos = bf16_tensor(cos4, device=submesh, mesh_axis=sp_axis, shard_dim=2)
    tt_sin = bf16_tensor(sin4, device=submesh, mesh_axis=sp_axis, shard_dim=2)
    tt_adaln = bf16_tensor(adaln_input, device=submesh)

    def run(logical_n: int):
        out = tt_block(
            tt_x,
            cos=tt_cos,
            sin=tt_sin,
            adaln_input=tt_adaln,
            attn_mask=None,
            spatial_sequence_length=logical_n,
        )
        ttnn.synchronize_device(submesh)
        ttnn.deallocate(out)

    local = padded_len // sp_factor  # per-device seq len
    _log(f"[hang-repro] image_len={image_len} padded_len={padded_len} local_seq={local} sp={sp_factor} tp={tp_factor}")

    # logical_n = image_len + n_text. Sweep n_text over values that straddle interesting
    # boundaries: chunk multiples, partial chunks, small tails, and cases where the real
    # tokens don't reach the later SP-ring participants (logical_n < (sp-1)*local -> some
    # ring device holds ZERO real tokens, a prime deadlock suspect).
    n_texts = [16, 64, 100, 128, 137, 256, 257, 384, 512, 777, 1024, 1536, 2000, 2048]
    logical_ns = [image_len + n for n in n_texts]

    # Warm the program cache with the FIRST logical_n (this call may JIT-compile; every
    # later call reuses the same program since shapes/program_config are constant).
    _log(f"[hang-repro] WARMUP logical_n={logical_ns[0]} (may compile)…")
    run(logical_ns[0])
    _log(f"[hang-repro] WARMUP done. Sweeping {len(logical_ns)} logical_n values (all warm).")

    for ln in logical_ns:
        _log(
            f"[hang-repro] BEGIN logical_n={ln} (n_text={ln - image_len}, local={local}, "
            f"real_ring_devices={min(sp_factor, -(-ln // local))}/{sp_factor})"
        )
        run(ln)
        _log(f"[hang-repro] END   logical_n={ln} OK")

    _log("[hang-repro] ALL logical_n values completed — NO HANG at this image_len.")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Full-pipeline repro: the block-level sweep above does NOT hang, so widen to the
# faithful server scenario — the real pipeline running SUCCESSIVE generations with
# DIFFERENT prompt lengths (untraced, 2048px), which is exactly what wedged the
# board (5 fine, then a new prompt hung). Each gen logs before/after with the
# tokenized length; a hang pinpoints the generation + its logical_n.
# ---------------------------------------------------------------------------

# Varied-length prompts. Order mirrors a real session: repeat one length (warm),
# then switch length several times to exercise the logical_n transition across calls.
_PROMPTS = [
    ("short_A", "A serene mountain lake at dawn, mist over the water, photorealistic."),
    ("short_A2", "A serene mountain lake at dawn, mist over the water, photorealistic."),
    (
        "medium_B",
        "A bustling neon-lit cyberpunk street market at night, dense crowds, rain-slicked pavement "
        "reflecting signage in Japanese and English, steam rising from food stalls, cinematic wide shot, "
        "shallow depth of field, ultra-detailed.",
    ),
    (
        "long_C",
        '{"high_level_description":"A polished high-resolution character reference sheet for Kaelyn, a '
        'glamorous adult woman in ornate fantasy armor","views":["front full body","three-quarter","back",'
        '"close-up face"],"style":"painterly concept art, dramatic rim lighting, muted teal and gold palette",'
        '"annotations":["material callouts","color swatches","weapon detail inset"],"typography":'
        '{"headline":"KAELYN","subtext":"Vanguard of the Ashen Reach"},"notes":"consistent face across all '
        'views, high detail on filigree, cohesive lighting, sheet layout on parchment background"}',
    ),
    ("short_A3", "A serene mountain lake at dawn, mist over the water, photorealistic."),
    (
        "medium_D",
        "An astronaut riding a horse across the surface of Mars, red dust kicked up, Earth visible in the "
        "sky, golden hour lighting, hyperrealistic, 85mm lens, dramatic composition.",
    ),
]


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [pytest.param((4, 2), (4, 2), 1, id="sp4tp2")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}], indirect=True
)
def test_pipeline_multi_prompt_hang(*, mesh_device, submesh_shape, tp_axis) -> None:
    from ....pipelines.ideogram4.pipeline import Ideogram4Pipeline

    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    _log("[hang-repro-pipe] building pipeline (SP4xTP2)…")
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)
    _log("[hang-repro-pipe] pipeline ready. Running successive generations (untraced, 2048px, TURBO_12).")

    for i, (name, prompt) in enumerate(_PROMPTS):
        n_tok = pipe.count_text_tokens(prompt)
        _log(f"[hang-repro-pipe] BEGIN gen {i + 1}/{len(_PROMPTS)} '{name}' n_text={n_tok}")
        img = pipe(prompt, height=2048, width=2048, preset="V4_TURBO_12", seed=1234, traced=False)
        _log(f"[hang-repro-pipe] END   gen {i + 1}/{len(_PROMPTS)} '{name}' ok std={img.std():.1f}")

    _log("[hang-repro-pipe] ALL generations completed — NO HANG.")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# SOAK: the 6-gen run above didn't hang, and the block-level sweep didn't either,
# so the hang looks intermittent. This loops MANY generations cycling prompt
# length AND preset (incl QUALITY_48, like the server did) to catch a rare wedge.
# Run under the tensix watcher (ETH disabled — the eth watcher overflows the
# fabric ERISC config buffer) + a hard timeout, so if it hangs we get:
#   * the last "SOAK BEGIN i"-without-"END" line -> the exact gen/preset/n_text,
#   * generated/watcher/watcher.log -> which tensix cores/kernels are stuck.
#
#   TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 SOAK_MAX_GENS=80 timeout 3600 \
#     pytest .../test_hang_ideogram4.py::test_pipeline_soak_hang -s -q \
#       -p no:cacheprovider --timeout=0
# ---------------------------------------------------------------------------

_SOAK_PRESETS = ["V4_TURBO_12", "V4_DEFAULT_20", "V4_QUALITY_48"]


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [pytest.param((4, 2), (4, 2), 1, id="sp4tp2")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}], indirect=True
)
def test_pipeline_soak_hang(*, mesh_device, submesh_shape, tp_axis) -> None:
    from ....pipelines.ideogram4.pipeline import Ideogram4Pipeline

    max_gens = int(os.environ.get("SOAK_MAX_GENS", "80"))
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    _log(f"[soak] building pipeline (SP4xTP2); will run up to {max_gens} gens…")
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)
    _log("[soak] pipeline ready. Cycling prompt-length x preset, untraced, 2048px.")

    for i in range(max_gens):
        name, prompt = _PROMPTS[i % len(_PROMPTS)]
        preset = _SOAK_PRESETS[i % len(_SOAK_PRESETS)]
        n_tok = pipe.count_text_tokens(prompt)
        _log(f"[soak] SOAK BEGIN {i + 1}/{max_gens} preset={preset} '{name}' n_text={n_tok}")
        t0 = time.time()
        img = pipe(prompt, height=2048, width=2048, preset=preset, seed=1000 + i, traced=False)
        _log(
            f"[soak] SOAK END   {i + 1}/{max_gens} preset={preset} '{name}' ok std={img.std():.1f} dt={time.time() - t0:.0f}s"
        )

    _log(f"[soak] ALL {max_gens} soak generations completed — NO HANG.")
    sys.stdout.flush()
