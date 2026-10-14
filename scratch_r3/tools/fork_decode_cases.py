# #58817 fork Auto TTSync audit: the decode matmuls Blackhole models run through the Metal 2.0 bmm fork, per device.
# tt_transformers models: every configuration comes from the model's own ModelArgs (dummy weights, the model_params
# config), built on a view of the one test device that reports the model's device count, so program configs, memory
# configs, dtypes and compute configs are the per-device ones of that mesh. Each matmul is built as attention.py and
# mlp.py call it in decode. qwen36 (P150x4): its 1D decode program configs (tt/tp_common.py) at the 27B per-device shapes.
import math
import os

import torch
import ttnn

BF, B8, B4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b


class DevView:
    """The test device, reporting a mesh of nd devices to ModelArgs (config only; tensors stay on the device)."""

    def __init__(self, dev, nd):
        self._dev, self._nd = dev, nd

    def get_num_devices(self):
        return self._nd

    @property
    def shape(self):
        return (1, self._nd)

    def __getattr__(self, name):
        return getattr(self._dev, name)


# (case prefix, HF name, devices per model, precision)
TT_MODELS = [
    ("l31_8b_p150_perf", "meta-llama/Llama-3.1-8B-Instruct", 1, "performance"),
    ("l31_8b_p150_acc", "meta-llama/Llama-3.1-8B-Instruct", 1, "accuracy"),
    ("l32_1b_p150_perf", "meta-llama/Llama-3.2-1B-Instruct", 1, "performance"),
    ("l32_3b_p150_perf", "meta-llama/Llama-3.2-3B-Instruct", 1, "performance"),
    ("mis_7b_p150_perf", "mistralai/Mistral-7B-Instruct-v0.3", 1, "performance"),
    ("l31_70b_p150x4_perf", "meta-llama/Llama-3.1-70B-Instruct", 4, "performance"),
    ("l31_70b_p150x4_acc", "meta-llama/Llama-3.1-70B-Instruct", 4, "accuracy"),
    ("q3_32b_p150x4_perf", "Qwen/Qwen3-32B", 4, "performance"),
]


def _shard_width(cores, rows, width):
    return ttnn.create_sharded_memory_config(
        (rows, width), cores, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True
    )


def tt_model_cases(dev, prefix, hf, nd, prec):
    """[(name, build)] where build() returns (in0, in1, kwargs for ttnn.linear)."""
    os.environ["HF_MODEL"] = hf
    from models.tt_transformers.tt.common import Mode
    from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, OpGroup, TensorGroup

    opt = (lambda a: DecodersPrecision.performance(a.n_layers, a.model_name)) if prec == "performance" else (
        lambda a: DecodersPrecision.accuracy(a.n_layers, a.model_name)
    )
    a = ModelArgs(DevView(dev, nd), dummy_weights=True, max_batch_size=32, max_seq_len=4096, optimizations=opt)
    a.n_layers = 1
    o = a.optimizations
    D = Mode.DECODE
    act = o.get_tensor_dtype(decoder_id=0, tensor=TensorGroup.ACTIVATION)
    dt = lambda g: o.get_tensor_dtype(decoder_id=0, tensor=g) or B8
    cfg = lambda g: o.get_math_fidelity(decoder_id=0, op=g, configuration=a)
    M = a.tile_padded_batch_rows
    grid = dev.compute_with_storage_grid_size()
    nlh = a.n_heads // nd
    specs = [
        # attention.py forward_decode: x is the attention norm's output in the attention input config
        ("qkv", a.dim, a.qkv_size // nd, BF, a.get_attn_input_mem_config(D), dt(TensorGroup.WQKV),
         dict(memory_config=a.get_attn_qkv_mm_mem_config(D), program_config=a.get_attn_qkv_program_config(D, 1),
              compute_kernel_config=cfg(OpGroup.LI_QKV_DECODE), dtype=act or BF)),
        # attention.py: wo after nlp_concat_heads_decode, one head per core
        ("wo", nlh * a.head_dim, a.dim, BF, _shard_width(ttnn.num_cores_to_corerangeset(nlh, grid, True), M, a.head_dim),
         dt(TensorGroup.WO),
         dict(memory_config=a.get_attn_wo_output_mem_config(D), program_config=a.get_attn_wo_program_config(D, 1),
              compute_kernel_config=cfg(OpGroup.LI_O_DECODE))),
        # mlp.py forward: w1 (w3 is the same matmul)
        ("ff1", a.dim, a.hidden_dim // nd, BF, a.get_mlp_input_mem_config(D), dt(TensorGroup.FF1_FF3),
         dict(memory_config=a.get_mlp_ff1_3_mem_config(D), program_config=a.get_mlp_ff1_3_prg_config(D, 1),
              compute_kernel_config=cfg(OpGroup.LI_FF1_FF3), dtype=act or BF)),
        # mlp.py: w2 on the product of w1 and w3 (activation dtype or bfp8), resharded for the DRAM-sharded matmul
        ("ff2", a.hidden_dim // nd, a.dim, act or B8, a.get_mlp_binary_mult_mem_config(D), dt(TensorGroup.FF2),
         dict(memory_config=a.get_mlp_ff2_mem_config(D), program_config=a.get_mlp_ff2_prg_config(D, 1),
              compute_kernel_config=cfg(OpGroup.LI_FF2), dtype=act or BF)),
    ]
    out = []
    for name, k, n, d0, mc0, d1, kw in specs:
        mc1 = a.create_dram_sharded_mem_config(k, n)

        def build(k=k, n=n, d0=d0, mc0=mc0, d1=d1, mc1=mc1, kw=kw):
            torch.manual_seed(0)
            x = ttnn.from_torch(torch.randn(1, 1, M, k), dtype=d0, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc0)
            w = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.02, dtype=d1, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc1)
            return x, w, kw

        out.append((f"{prefix}_{name}", build, (k, n, str(kw["program_config"]).replace("\n", " "))))
    return out


def qwen36_cases(dev):
    """qwen36 decode on P150x4, per device (27B: dim 5120, gate/up N 4352 per tp_common.py:284, attention and GDN
    out 1536 = 24 heads x 256 / 4 = 48 value heads x 128 / 4, fused QKV+gate 3584 = 6 x 256 x 2 + 2 x 256)."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    gw = dev.compute_with_storage_grid_size().x
    lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
    hifi2 = tpc.COMPUTE_HIFI2
    specs = [
        # mlp.py: compute_kernel_config_decode, bfp4 gate/up, bfp8 down; model_config.py:236-251
        ("q36_27b_gate", 5120, 4352, B4, lofi, tpc.create_matmul_1d_decode_progcfg(32, 5120, 4352, num_cores=44, fused_activation=ttnn.UnaryOpType.SILU, grid_w=gw)),
        ("q36_27b_up", 5120, 4352, B4, lofi, tpc.create_matmul_1d_decode_progcfg(32, 5120, 4352, num_cores=44, grid_w=gw)),
        ("q36_27b_down", 4352, 5120, B8, lofi, tpc.create_matmul_1d_decode_progcfg(32, 4352, 5120, num_cores=33, grid_w=gw)),
        # attention/tp.py and gdn/tp.py: COMPUTE_HIFI2, bfp8 weights; model_config.py:256-282
        ("q36_27b_attn_qkv", 5120, 3584, B8, hifi2, tpc.create_matmul_1d_decode_progcfg(32, 5120, 3584, num_cores=64)),
        ("q36_27b_attn_wo", 1536, 5120, B8, hifi2, tpc.create_matmul_1d_decode_progcfg(32, 1536, 5120, num_cores=33, grid_w=gw)),
    ]
    out = []
    for name, k, n, d1, ckc, pc in specs:
        def build(k=k, n=n, d1=d1, ckc=ckc, pc=pc):
            torch.manual_seed(0)
            x = ttnn.from_torch(torch.randn(1, 1, 32, k), dtype=BF, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
            w = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.02, dtype=d1, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return x, w, dict(program_config=pc, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)

        out.append((name, build, (k, n, str(pc).replace("\n", " "))))
    return out
