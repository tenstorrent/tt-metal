# Compare TTNN RelPositionMultiHeadAttention with NeMo's using real weights from parakeet-tdt-0.6b-v2

import torch
import ttnn
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention
from models.experimental.parakeet.tt.ttnn_conf_layer import TtRelPositionMultiHeadAttention

# ----------------------------------------------------------------------
# 1) Load NeMo model and extract RelPositionMultiHeadAttention from a layer
# ----------------------------------------------------------------------
model_name = "nvidia/parakeet-tdt-0.6b-v2"
print(f"Loading {model_name} ...")
asr_model = EncDecRNNTBPEModel.from_pretrained(model_name, map_location="cpu")
asr_model.eval()

layer_idx = 0
encoder_layer = asr_model.encoder.layers[layer_idx]
attn = encoder_layer.self_attn  # NeMo RelPositionMultiHeadAttention instance

# Extract weights and biases
q_w = attn.linear_q.weight.detach()  # [n_feat, n_feat]
k_w = attn.linear_k.weight.detach()
v_w = attn.linear_v.weight.detach()
out_w = attn.linear_out.weight.detach()
linear_pos_w = attn.linear_pos.weight.detach()  # [n_feat, n_feat]
pos_bias_u = attn.pos_bias_u.detach()  # [n_head, head_dim]
pos_bias_v = attn.pos_bias_v.detach()

q_bias = attn.linear_q.bias.detach() if attn.linear_q.bias is not None else None
k_bias = attn.linear_k.bias.detach() if attn.linear_k.bias is not None else None
v_bias = attn.linear_v.bias.detach() if attn.linear_v.bias is not None else None
out_bias = attn.linear_out.bias.detach() if attn.linear_out.bias is not None else None

# ----------------------------------------------------------------------
# 2) Prepare dummy inputs
# ----------------------------------------------------------------------
n_feat = attn.linear_q.in_features
n_head = attn.h
head_dim = n_feat // n_head
B, T = 2, 32

x = torch.randn(B, T, n_feat)
# NeMo's RelPositionMultiHeadAttention expects pos_emb shape (batch, 2*T-1, n_feat)
pos_emb = torch.randn(B, 2 * T - 1, n_feat)
mask = None  # or torch.zeros(B, T, T, dtype=torch.bool)

# ----------------------------------------------------------------------
# 3) Run NeMo RelPositionMultiHeadAttention
# ----------------------------------------------------------------------
with torch.no_grad():
    nemo_out = RelPositionMultiHeadAttention(query=x, key=x, value=x, mask=mask, pos_emb=pos_emb)


device = ttnn.open_device(device_id=0)
dtype = ttnn.bfloat16
compute_cfg = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)

tt_attn = TtRelPositionMultiHeadAttention(
    n_head=n_head,
    n_feat=n_feat,
    dropout_rate=0.0,
    pos_bias_u=pos_bias_u,
    pos_bias_v=pos_bias_v,
    device=device,
    dtype=dtype,
    compute_kernel_config=compute_cfg,
)


def to_ttnn(t, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.to(dtype=torch.bfloat16), layout=layout, device=device)


# Fuse QKV weights in the order expected by TTNN (Q, K, V concatenated)
qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)  # [3*n_feat, n_feat]
qkv_bias = None
if q_bias is not None and k_bias is not None and v_bias is not None:
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

tt_attn.set_weights(
    linear_pos_weight=to_ttnn(linear_pos_w, layout=ttnn.TILE_LAYOUT),
    qkv_weight=to_ttnn(qkv_weight, layout=ttnn.TILE_LAYOUT),
    out_weight=to_ttnn(out_w, layout=ttnn.TILE_LAYOUT),
    qkv_bias=to_ttnn(qkv_bias, layout=ttnn.TILE_LAYOUT) if qkv_bias is not None else None,
    out_bias=to_ttnn(out_bias, layout=ttnn.TILE_LAYOUT) if out_bias is not None else None,
)

x_tt = to_ttnn(x, layout=ttnn.TILE_LAYOUT)
pos_emb_tt = to_ttnn(pos_emb, layout=ttnn.TILE_LAYOUT)
mask_tt = to_ttnn(mask, layout=ttnn.TILE_LAYOUT) if mask is not None else None

with torch.no_grad():
    tt_out_tt = tt_attn.forward(query=x_tt, key=x_tt, value=x_tt, mask=mask_tt, pos_emb=pos_emb_tt)
    tt_out = ttnn.to_torch(tt_out_tt)

# ----------------------------------------------------------------------
# 6) Compare outputs
# ----------------------------------------------------------------------

print("\n📊 Output Metrics:")
pcc = compute_pcc(tt_out, nemo_out)
max_err = compute_max_abs_error(tt_out, nemo_out)
mean_err = compute_mean_abs_error(tt_out, nemo_out)

print(f"PCC:            {pcc:.6f}")
print(f"Max Abs Error:  {max_err:.6f}")
print(f"Mean Abs Error: {mean_err:.6f}")
print("NeMo output shape:", nemo_out.shape)
print("TTNN output shape:", tt_out.shape)

# Cleanup
ttnn.close_device(device)
