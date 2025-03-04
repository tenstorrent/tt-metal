from loguru import logger
import torch
import ttnn

from models.demos.t3000.llama2_70b.reference.llama.llama.model import precompute_freqs_cis, apply_rotary_emb
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import skip_for_grayskull, skip_for_blackhole, nearest_32, skip_for_wormhole_b0
from models.demos.llama3.tt.llama_common import (
    precompute_freqs,
    get_rot_transformation_mat,
)
from models.demos.llama3.tt.llama_rope import TtLlamaRotarySetup


def compute_gather_cos_sin(dhead, end, position_ids):
    cos, sin = precompute_freqs(
        dhead, end, theta=10000.0, scale_factor=None, orig_context_len=131072
    )  # Using reference defaults (no scaling)
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


class TtLlamaRotary(torch.nn.Module):
    def __init__(self, device, head_dim: int, datatype=ttnn.bfloat16):
        super().__init__()

        self.head_dim = head_dim
        self.device = device

        self.transformation_mat = ttnn.from_torch(
            get_rot_transformation_mat(dhead=ttnn.TILE_SIZE), device=device, layout=ttnn.TILE_LAYOUT, dtype=datatype
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            # math_fidelity=ttnn.MathFidelity.LoFi,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=(True if self.head_dim <= 128 else False),
            packer_l1_acc=True,
        )

    def apply_rotary(self, x, cos, sin):
        # n_head = 8 for Q
        # n_head = 1 for K

        rotary_output = ttnn.experimental.rotary_embedding_llama(
            x, cos, sin, self.transformation_mat, is_decode_mode=False
        )

        return rotary_output

    def apply_fused_rotary(self, q, k, cos, sin):
        # n_head = 8 for Q
        # n_head = 1 for K
        rotary_output_q, rotary_output_k = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q,
            k,
            cos,
            sin,
            self.transformation_mat,
            compute_kernel_config=self.compute_kernel_config,
        )

        return rotary_output_q, rotary_output_k

    def forward(self, xq, cos, sin):
        xq = self.apply_rotary(xq, cos, sin)
        return xq


def dump_xtensor(name, x):
    try:
        l = x.tolist()
    except Exception:
        l = x.item()  # For scalar tensors

    def recursive_format(x):
        if isinstance(x, list):
            return "{" + ", ".join(recursive_format(item) for item in x) + "}"
        else:
            return f"{float(x):.5f}F"

    tensor_str = recursive_format(l)
    return f"xt::xarray<float> {name} = {tensor_str};"


def rope_test(device):
    batch = 1
    n_heads = 2
    seq_len = 5
    head_dim = 32
    torch.manual_seed(42)

    def on_l1(x):
        return ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    xq = ttnn.from_torch(
        torch.ones(batch, n_heads, seq_len, head_dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    cos, sin = compute_gather_cos_sin(
        dhead=head_dim,
        end=seq_len,
        position_ids=torch.arange(0, seq_len),
    )
    # print(dump_xtensor("cos", cos))
    # print(dump_xtensor("sin", sin))
    tt_model = TtLlamaRotary(device, head_dim, ttnn.bfloat16)
    # trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)
    # print(dump_xtensor("expected_trans_mat", trans_mat))
    tt_out = tt_model(xq, on_l1(cos), on_l1(sin))
    print(dump_xtensor("expected_out", tt_out.cpu().to_torch()))


rope_test(ttnn.open_device(device_id=0))
