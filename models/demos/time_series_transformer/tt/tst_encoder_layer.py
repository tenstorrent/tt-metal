# SPDX-License-Identifier: Apache-2.0
import torch, ttnn

D_MODEL = 26
NUM_HEADS = 2
HEAD_DIM = D_MODEL // NUM_HEADS


def _tw(w):
    """Weight matrix: was stored transposed for TTNN, un-transpose for torch matmul."""
    return ttnn.to_torch(w).float().T


def _tb(w, size=None):
    """Bias vector."""
    t = ttnn.to_torch(w).float()
    return t[:size] if size else t


def _attn(h, kv, w, pfx):
    B, tgt = h.shape[:2]; src = kv.shape[1]
    q = (h  @ _tw(w[pfx+".q_proj.weight"])  + _tb(w[pfx+".q_proj.bias"],  D_MODEL)).view(B,tgt,NUM_HEADS,HEAD_DIM).transpose(1,2)
    k = (kv @ _tw(w[pfx+".k_proj.weight"])  + _tb(w[pfx+".k_proj.bias"],  D_MODEL)).view(B,src,NUM_HEADS,HEAD_DIM).transpose(1,2)
    v = (kv @ _tw(w[pfx+".v_proj.weight"])  + _tb(w[pfx+".v_proj.bias"],  D_MODEL)).view(B,src,NUM_HEADS,HEAD_DIM).transpose(1,2)
    a = torch.softmax((q @ k.transpose(-2,-1)) * HEAD_DIM**-0.5, dim=-1)
    out = (a @ v).transpose(1,2).contiguous().view(B, tgt, D_MODEL)
    return out @ _tw(w[pfx+".out_proj.weight"]) + _tb(w[pfx+".out_proj.bias"], D_MODEL)


def _ln(x, w, pfx):
    wt = _tb(w[pfx+".weight"], D_MODEL)
    wb = _tb(w[pfx+".bias"],   D_MODEL)
    return torch.nn.functional.layer_norm(x, [D_MODEL], wt, wb)


def tst_encoder_layer(device, hidden_states, weights, layer_idx):
    w = weights[f"encoder.layers.{layer_idx}"]
    h = ttnn.to_torch(hidden_states).float()[..., :D_MODEL]

    h = _ln(h + _attn(h, h, w, "self_attn"), w, "self_attn_layer_norm")

    fc1w = _tw(w["fc1.weight"]); fc1b = _tb(w["fc1.bias"])
    fc2w = _tw(w["fc2.weight"]); fc2b = _tb(w["fc2.bias"], D_MODEL)
    ffn = torch.nn.functional.gelu(h @ fc1w + fc1b)
    h = _ln(h + (ffn @ fc2w)[..., :D_MODEL] + fc2b, w, "final_layer_norm")

    out = torch.nn.functional.pad(h, (0, (-D_MODEL) % 32))
    return ttnn.from_torch(out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
