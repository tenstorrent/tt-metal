"""Experiment A — ISOLATED sparse_sdpa op vs the vLLM GPU trace.

Feed the trace's REAL absorbed inputs (q576, kvpe576, indices2048) for a layer into our ttnn
`ops.sparse_mla`, compare our [.,64,.,512] output to the trace's `sparse_sdpa_output` directly.
Each layer is independent (teacher-forced from the trace) — pure op fidelity, no accumulation.

scale = 0.0625 (= qk_head_dim^-0.5 = 256^-0.5; GLM has no YaRN/mscale). Sentinel -1 == 0xFFFFFFFF.
Run:  pytest models/demos/deepseek_v32/tests/test_sparse_sdpa_vs_gpu.py -k sp4xtp2 -s
"""
import glob

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v32.tests.mesh_utils import parametrize_mesh_device
from models.demos.deepseek_v32.tt import ops

TRACE = "/localdev/nmilicevic/tt-metal/bit_sculpt/results/glm-51-traces/vllm-glm51-sdpa-5k-trace"
SCALE = 0.0625
SEQ = 5120
HEADS = 64
LAYERS = [0, 3, 15, 30, 45, 60, 77]  # representative depth slice (extend to range(78) once all pulled)


def _stream(rel):
    fs = [f for f in glob.glob(f"{TRACE}/{rel}/*.safetensors") if __import__("os").path.getsize(f) >= 1024]
    if not fs:
        return None
    return next(iter(load_file(fs[0]).values()))


@parametrize_mesh_device()
@pytest.mark.parametrize("layer", LAYERS, ids=[f"L{l}" for l in LAYERS])
@pytest.mark.timeout(0)
def test_sparse_sdpa_op_vs_trace(mesh_device, layer):
    q = _stream(f"sparse_sdpa/sparse_sdpa_input_layer_{layer}")  # (5120, 64, 576)
    kvpe = _stream(f"kv_cache/layer_{layer}")  # (5120, 576) — dir is kv_cache/layer_{i}, key=kv_post_transform
    idx = _stream(f"dsa/dsa_topk_indices_layer_{layer}")  # (5120, 2048) int32, -1 sentinel
    ref = _stream(f"sparse_sdpa/sparse_sdpa_output_layer_{layer}")  # (5120, 64, 512)
    if any(t is None for t in (q, kvpe, idx, ref)):
        pytest.skip(f"layer {layer}: trace stream(s) not pulled yet (git lfs pull pending)")

    # RoPE-FRAME FIX: the trace stores q_pe and k_pe in opposite rope layouts (half-split vs
    # interleaved). q·kvpe is only correct when both share a frame, so re-order kvpe's 64 rope dims
    # (512:576) half-split->interleaved to match q. V (kvpe[:512]) is untouched. (Verified exact: PCC 1.0.)
    _h2i = torch.empty(64, dtype=torch.long)
    _h2i[0::2] = torch.arange(32)
    _h2i[1::2] = torch.arange(32, 64)
    kvpe = kvpe.clone()
    kvpe[:, 512:576] = kvpe[:, 512:576][:, _h2i]

    sp, tp = mesh_device.shape
    k = idx.shape[-1]
    # q: (S,64,576) -> [1,64,S,576]; SP-shard seq(dim2,axis0) + TP-shard heads(dim1,axis1) (matches the op layout).
    q_t = q.reshape(SEQ, HEADS, 576).permute(1, 0, 2).unsqueeze(0).contiguous().to(torch.bfloat16)
    q_dev = ttnn.from_torch(
        q_t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, 1)),
    )
    # kvpe: (S,576) -> [1,1,S,576] replicated, ROW_MAJOR bf16 (full-T latent prefix).
    kvpe_dev = ttnn.from_torch(
        kvpe.reshape(1, 1, SEQ, 576).contiguous().to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    # indices: (S,k) -> [1,1,S,k] uint32 (int32 -1 -> 0xFFFFFFFF via bit reinterpret), replicated.
    idx_u32 = idx.reshape(1, 1, SEQ, k).contiguous().to(torch.int32).view(torch.uint32)
    idx_dev = ttnn.from_torch(
        idx_u32,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out = ops.sparse_mla(q_dev, kvpe_dev, idx_dev, SCALE, start_pos=0, sp_axis=0, tp_axis=1)
    out_t = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=(sp, tp))
    )  # [1, 64, S, 512]
    ours = out_t[0].permute(1, 0, 2).contiguous()  # (S, 64, 512)

    _, pcc = comp_pcc(ref.float(), ours.float(), 0)
    logger.info(f">>> [Experiment A] L{layer}: sparse_sdpa op vs vLLM trace  PCC = {pcc}")
    assert pcc >= 0.99, f"L{layer} sparse_sdpa op vs trace PCC {pcc} < 0.99"
