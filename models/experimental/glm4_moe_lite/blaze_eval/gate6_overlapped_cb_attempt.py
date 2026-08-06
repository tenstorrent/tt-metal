import sys

sys.path.insert(0, "/home/ttuser/sdawle/skills/blaze-vs-ttnn-bench/scripts")
sys.path.insert(0, "/home/ttuser/sdawle/tt-blaze/tests/blaze/micro_ops/common")
import torch, ttnn, ab_harness
from blaze.fused_program import FusedProgram
from blaze.ops.glm_qkv_a_projection import GLMQKVAProjection
from blaze.ops.dram_streaming_matmul import DRAMStreamingMatmul
from blaze.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment
from test_dram_streaming_matmul import _make_output_tensor, _make_weights_tensor, _pad_to_dram_banks, _roundtrip_weights
from models.common.utility_functions import comp_pcc

d = ttnn.open_device(device_id=0)
K, NQ, NKV, TW = 2048, 768, 576, 32
cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(d, ttnn.NOC.NOC_0)
banks = len(cores)
crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
nq, nkv = _pad_to_dram_banks(NQ, TW, TW * banks), _pad_to_dram_banks(NKV, TW, TW * banks)
torch.manual_seed(0)
row = torch.randn(1, 1, 1, K).bfloat16().float()
wq, wkv = torch.randn(1, 1, K, nq).bfloat16().float(), torch.randn(1, 1, K, nkv).bfloat16().float()
gq = DRAMStreamingMatmul.golden(row, _roundtrip_weights(wq, ttnn.bfloat8_b))
gkv = DRAMStreamingMatmul.golden(row, _roundtrip_weights(wkv, ttnn.bfloat8_b))
b_wq = _make_weights_tensor(d, wq, k=K, n_padded=nq, tile_w=TW, num_banks=banks, weight_dtype=ttnn.bfloat8_b)
b_wkv = _make_weights_tensor(d, wkv, k=K, n_padded=nkv, tile_w=TW, num_banks=banks, weight_dtype=ttnn.bfloat8_b)
oq = _make_output_tensor(d, m=1, n_padded=nq, tile_h=1, tile_w=TW, compute_core_grid=crs, per_core_N=nq // banks)
okv = _make_output_tensor(d, m=1, n_padded=nkv, tile_h=1, tile_w=TW, compute_core_grid=crs, per_core_N=nkv // banks)
x_model = ttnn.from_torch(row.repeat(1, 1, 32, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
# Keep the ORIGINAL (1,K) shard -- the (64,32) variant is what hung. No reshape in the chain.
act_mc = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(crs, (1, K), ttnn.ShardOrientation.ROW_MAJOR),
)
TILE_1x32, PAGE, TOTAL = ttnn.Tile([1, TW]), TW * 2, K * 2


def build_act():
    r = ttnn.slice(x_model, [0, 0, 0, 0], [1, 1, 1, K])
    rm = ttnn.to_layout(r, ttnn.ROW_MAJOR_LAYOUT)
    rep = ttnn.repeat(rm, ttnn.Shape([1, 1, banks, 1]))
    return ttnn.to_memory_config(rep, act_mc)


def run(act_t):
    f = FusedProgram(
        kernel=None,
        device=d,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        name="gate6",
    )
    # cb_from_tensor has no total_size; the OVERLAPPED variant takes it as a first-class arg,
    # which is what makes n_pages = total_size/page_size = 64 instead of the shard row count.
    act_cb = f.cb_from_tensor_overlapped(act_t, 0, TOTAL, PAGE, tile=TILE_1x32)
    GLMQKVAProjection.emit(
        f,
        act_cb,
        b_wq,
        b_wkv,
        q_a_out=oq,
        kv_a_out=okv,
        prefix="qkv_a",
        fp32_dest_acc_en=True,
        q_a_subblock_k=K // TW // 2,
        kv_a_subblock_k=K // TW // 2,
    )
    f.run()
    return oq, okv


try:
    run(build_act())
    pq = comp_pcc(gq, ttnn.to_torch(oq).float())[1]
    pkv = comp_pcc(gkv, ttnn.to_torch(okv).float())[1]
    print(f"RESULT PCC q_a={pq:.4f} kv_a={pkv:.4f}", flush=True)
    tot, _ = ab_harness._measure(d, lambda: run(build_act()), warmup=2, iters=5)
    conv, _ = ab_harness._measure(d, build_act, warmup=2, iters=5)
    print(f"RESULT cluster={tot:.1f}us (reshard {conv:.1f}) vs ttnn fused q_kv_a 45.1us -> {45.1/tot:.2f}x", flush=True)
except Exception as e:
    print("RESULT FAIL:", str(e).split(chr(10))[0][:140], flush=True)
ttnn.close_device(d)
