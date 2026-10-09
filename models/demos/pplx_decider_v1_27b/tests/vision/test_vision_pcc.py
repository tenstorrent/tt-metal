# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vision tower vs the HF bf16 golden, real weights, all 8 golden images (256..1024 patches).

- ``test_patch_embed``: Conv3d-as-matmul alone and + pos embed (block-0 input), >= 0.995;
- ``test_blocks_teacher_forced``: every one of the 27 blocks fed the golden input of that block, >= 0.995;
- ``test_merger``: merger fed the golden last hidden state, >= 0.995;
- ``test_tower_e2e``: the whole tower from pixels only, image features >= 0.99; the per-block PCC of the
  free-running hidden state is logged as ``tower_e2e_block`` (drift diagnostic, not gated);
- ``test_key_mask``: v02 (936 patches in the 1024 bucket) with the mask vs without (``cu = [0, S]``),
  and v01 (256 patches) run in the 512 bucket, to show the padded keys are masked.
- ``test_gelu_variant``: the fused ``"gelu_tanh"`` activation is the tanh GELU (not erf), and ``"gelu"`` erf.

Every PCC goes to ``$PPLX_DECIDER_VISION_PCC_LOG`` (stage12a/pcc_vision.jsonl).
"""

import pytest
import torch

import ttnn
from models.demos.pplx_decider_v1_27b.tests.vision.vision_test_utils import (
    DEVICE_PARAMS,
    IMAGES,
    TOWER_THRESHOLD,
    block_input,
    build_tower,
    golden_tower,
    image_ids,
    pcc_record,
    prepare,
    to_host_rows,
)

pytestmark = pytest.mark.use_module_device(DEVICE_PARAMS)


@pytest.fixture(scope="module")
def tower(_device_module_impl):
    return build_tower(_device_module_impl)


def _assert_all(records):
    bad = [f"{r['module']} B{r['block']} {r['image']}: {r['pcc']:.6f}" for r in records if not r["passed"]]
    assert not bad, f"{len(bad)} below threshold: {bad}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("image", IMAGES, ids=image_ids())
def test_patch_embed(tower, image):
    golden = golden_tower(image)
    inputs = prepare(tower, image)
    n = inputs.num_patches
    emb = tower.patch_embed.embed(inputs.pixels)
    full = tower.embed(inputs)
    records = [
        pcc_record(
            golden["patch_embed"], to_host_rows(emb, n), module="patch_embed", image=image, bucket=inputs.bucket
        ),
        pcc_record(
            golden["embed_with_pos"], to_host_rows(full, n), module="patch_embed_pos", image=image, bucket=inputs.bucket
        ),
    ]
    inputs.deallocate()
    _assert_all(records)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("image", IMAGES, ids=image_ids())
def test_blocks_teacher_forced(tower, image):
    golden = golden_tower(image)
    inputs = prepare(tower, image)
    n, s = inputs.num_patches, inputs.bucket
    records = []
    for i in range(len(tower.blocks)):
        x = tower.upload_hidden(block_input(golden, i), s)
        out = tower.run_block(i, x, inputs)
        records.append(
            pcc_record(golden[f"block_{i:02d}"], to_host_rows(out, n), module="block", block=i, image=image, bucket=s)
        )
        ttnn.deallocate(x)
        ttnn.deallocate(out)
    inputs.deallocate()
    _assert_all(records)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("image", IMAGES, ids=image_ids())
def test_merger(tower, image):
    golden = golden_tower(image)
    n = golden["block_26"].shape[0]
    from models.demos.pplx_decider_v1_27b.tt.vision.config import vision_bucket_for

    s = vision_bucket_for(n)
    x = tower.upload_hidden(golden["block_26"], s)
    out = tower.merger(x)
    record = pcc_record(golden["merger"], to_host_rows(out, n // 4), module="merger", image=image, bucket=s)
    _assert_all([record])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("image", IMAGES, ids=image_ids())
def test_tower_e2e(tower, image):
    golden = golden_tower(image)
    inputs = prepare(tower, image)
    n, s = inputs.num_patches, inputs.bucket
    # Drift diagnostic: the free-running hidden state after every block (not gated).
    x = tower.embed(inputs)
    for i in range(len(tower.blocks)):
        nxt = tower.run_block(i, x, inputs)
        ttnn.deallocate(x)
        x = nxt
        pcc_record(
            golden[f"block_{i:02d}"],
            to_host_rows(x, n),
            module="tower_e2e_block",
            block=i,
            image=image,
            bucket=s,
            threshold=TOWER_THRESHOLD,
        )
    ttnn.deallocate(x)
    # Gate: the public forward (pixels -> sliced features).
    features = tower(inputs)
    assert tuple(features.shape) == (1, 1, n // 4, 5120), features.shape
    record = pcc_record(
        golden["merger"],
        to_host_rows(features, n // 4),
        module="tower_e2e",
        image=image,
        bucket=s,
        threshold=TOWER_THRESHOLD,
    )
    inputs.deallocate()
    _assert_all([record])


@pytest.mark.timeout(1800)
def test_key_mask(tower):
    records = []
    # v02: 936 real patches in the 1024 bucket; with vs without the [0, n, S] window mask.
    golden = golden_tower("v02_count_circles")
    inputs = prepare(tower, "v02_count_circles")
    records.append(
        pcc_record(golden["merger"], to_host_rows(tower(inputs), 234), module="mask_on_v02", image="v02_count_circles")
    )
    unmasked = ttnn.from_torch(
        torch.tensor([0, inputs.bucket], dtype=torch.int32), device=tower.mesh_device, dtype=ttnn.int32
    )
    masked = inputs.cu_window_seqlens
    inputs.cu_window_seqlens = unmasked
    no_mask = pcc_record(
        golden["merger"],
        to_host_rows(tower(inputs), 234),
        module="mask_off_v02",
        image="v02_count_circles",
        threshold=0.0,
        extra={"note": "negative control: padded keys visible"},
    )
    inputs.cu_window_seqlens = masked
    inputs.deallocate()
    # v01: 256 patches forced into the 512 bucket (256 padded keys).
    golden = golden_tower("v01_dominant_color")
    inputs = prepare(tower, "v01_dominant_color", bucket=512)
    records.append(
        pcc_record(
            golden["merger"],
            to_host_rows(tower(inputs), 64),
            module="mask_on_v01_in_512",
            image="v01_dominant_color",
            bucket=512,
            threshold=TOWER_THRESHOLD,
        )
    )
    inputs.deallocate()
    _assert_all(records)
    assert no_mask["pcc"] < records[0]["pcc"], "unmasked padding should hurt; the mask must be what makes v02 exact"


@pytest.mark.timeout(600)
def test_gelu_variant(device):
    """Fused matmul activation on an identity weight, fp32 output: which GELU does each string compute?"""
    torch.manual_seed(0)
    x = torch.linspace(-5, 5, 64 * 128).reshape(64, 128).to(torch.bfloat16).float()
    eye = torch.eye(128)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    grid = device.compute_with_storage_grid_size()
    tx = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tw = ttnn.from_torch(eye, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    errs = {}
    for act in ("gelu_tanh", "gelu"):
        y = ttnn.to_torch(
            ttnn.linear(
                tx,
                tw,
                activation=act,
                dtype=ttnn.float32,
                compute_kernel_config=cfg,
                core_grid=ttnn.CoreGrid(x=grid.x, y=grid.y),
            )
        ).float()
        errs[act] = {
            ref: float((y - torch.nn.functional.gelu(x, approximate=ref)).abs().max()) for ref in ("tanh", "none")
        }
    pcc_record(
        x,
        x,
        module="gelu_variant_probe",
        image="synthetic",
        threshold=0.0,
        extra={"max_abs_err": errs},
    )
    assert errs["gelu_tanh"]["tanh"] < errs["gelu_tanh"]["none"], errs
    assert errs["gelu"]["none"] < errs["gelu"]["tanh"], errs
