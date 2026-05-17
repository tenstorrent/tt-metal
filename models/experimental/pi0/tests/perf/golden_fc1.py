"""Torch reference for SigLIP layer-0 FC1 matmul on real π0.5 weights.

FC1 shape: (1152, 4304). Padded to (1152, 4320) with 16 zero cols for tile
alignment. Real activation = patch_embed + position_embed output (same as
QKV golden — distribution-realistic for an MLP FC1 input).

Note: in deployment, FC1 input is layer-0 *attention-output + residual + LN2*
not the raw patch_embed output. Using patch_embed for now keeps the test
simple; the PCC check validates matmul mechanics regardless of activation
provenance. A separate test can run torch attention forward to get the actual
FC1 input distribution.
"""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

PI05_WEIGHTS = "/storage/sdawle/pi05_weights/pi05_base/model.safetensors"
VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower."

LAYER_IDX = 0
HIDDEN = 1152
INTERMEDIATE_LOGICAL = 4304
INTERMEDIATE_PADDED = 4320  # next multiple of 32
PATCH_SIZE = 14
IMAGE_SIZE = 224
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256
M = NUM_PATCHES
K = HIDDEN
N_LOGICAL = INTERMEDIATE_LOGICAL
N_PADDED = INTERMEDIATE_PADDED


def _load_vision_keys() -> dict:
    sd = load_file(PI05_WEIGHTS)
    return {k[len(VISION_PREFIX) :]: v for k, v in sd.items() if k.startswith(VISION_PREFIX)}


def load_layer0_fc1() -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (W_fc1 [K=1152, N=4304], b_fc1 [N=4304]).

    HF format is `(out, in)`; transpose to `(in, out)` for X @ W.
    """
    vw = _load_vision_keys()
    w = vw[f"vision_model.encoder.layers.{LAYER_IDX}.mlp.fc1.weight"]  # (4304, 1152)
    b = vw[f"vision_model.encoder.layers.{LAYER_IDX}.mlp.fc1.bias"]  # (4304,)
    w_in_out = w.T.contiguous()  # (1152, 4304)
    return w_in_out, b


def load_layer0_fc1_padded() -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (W_fc1_padded [K=1152, N=4320], b_fc1_padded [N=4320]).

    Pads the logical 4304 cols up to 4320 with zeros so the kernel can run
    on tile-aligned 135 N-tiles (27 cores × 5 tiles each).
    """
    w, b = load_layer0_fc1()
    pad_n = N_PADDED - N_LOGICAL  # 16
    w_padded = torch.cat([w, torch.zeros(K, pad_n, dtype=w.dtype)], dim=1).contiguous()  # (1152, 4320)
    b_padded = torch.cat([b, torch.zeros(pad_n, dtype=b.dtype)]).contiguous()  # (4320,)
    return w_padded, b_padded


def make_real_activation(seed: int = 42, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Real patch_embed + pos_embed output, same as QKV golden."""
    vw = _load_vision_keys()
    pe_w = vw["vision_model.embeddings.patch_embedding.weight"].float()
    pe_b = vw["vision_model.embeddings.patch_embedding.bias"].float()
    pos_emb = vw["vision_model.embeddings.position_embedding.weight"].float()

    g = torch.Generator().manual_seed(seed)
    pixel = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g, dtype=torch.float32)
    patches = F.conv2d(pixel, pe_w, bias=pe_b, stride=PATCH_SIZE)
    embeds = patches.flatten(2).transpose(1, 2) + pos_emb.unsqueeze(0)
    return embeds.squeeze(0).to(dtype)


def golden_fc1(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference FC1: y = x @ W + b. Returns (M, N_LOGICAL) bf16.

    Uses the **logical** (unpadded) W and b. PCC compare must trim kernel output
    to the first N_LOGICAL cols.
    """
    assert x.shape == (M, K), f"x shape {x.shape}"
    assert w.shape == (K, N_LOGICAL), f"w shape {w.shape}"
    assert b.shape == (N_LOGICAL,), f"b shape {b.shape}"
    y = (x.float() @ w.float()) + b.float()
    return y.to(x.dtype)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.flatten().float(), b.flatten().float()
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(), y.std()
    if sx < 1e-6 or sy < 1e-6:
        return 1.0 if torch.allclose(x, y, atol=1e-5) else 0.0
    return (((x - mx) * (y - my)).mean() / (sx * sy)).item()


if __name__ == "__main__":
    w, b = load_layer0_fc1()
    w_p, b_p = load_layer0_fc1_padded()
    x = make_real_activation()
    print(f"W FC1 logical : {tuple(w.shape)}  dtype={w.dtype}")
    print(f"W FC1 padded  : {tuple(w_p.shape)}  dtype={w_p.dtype}")
    print(f"b FC1 logical : {tuple(b.shape)}")
    print(f"b FC1 padded  : {tuple(b_p.shape)}")
    print(f"X activation  : {tuple(x.shape)}  dtype={x.dtype}")
    y_no_bias = (x.float() @ w.float()).to(x.dtype)
    print(f"Y no-bias     : {tuple(y_no_bias.shape)}  dtype={y_no_bias.dtype}")
    y_bias = golden_fc1(x, w, b)
    print(f"Y golden+bias : {tuple(y_bias.shape)}  dtype={y_bias.dtype}")
