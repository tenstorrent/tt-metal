"""Recipe library — workaround patterns the LLM can paste into its
ttnn code when a constraint fires.

Each recipe is a chunk of markdown with a short title, the WHY (what
constraint it works around), and a CODE TEMPLATE the LLM can adapt.
Templates use ``{placeholder}`` substitution against ``Violation.details``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Recipe:
    """A workaround pattern keyed by ``recipe_id``."""

    id: str
    title: str
    body: str  # markdown, may contain `{placeholder}` slots


RECIPES: Dict[str, Recipe] = {}


def _register(recipe: Recipe) -> None:
    RECIPES[recipe.id] = recipe


_register(
    Recipe(
        id="small_dim_layer_norm",
        title="Small-dim LayerNorm — ttnn.sum + manual divide (DO NOT use ttnn.mean)",
        body=(
            "Constraint: the dim being normalized has size {small_dim} (< tile_width "
            "{tile_width}). Input shape: {shape}.\n\n"
            "⚠ CRITICAL — ttnn.mean is padding-poisoned on sub-tile dims:\n"
            "  `ttnn.mean(x, dim=-1)` internally pads the reduction dim to tile_width=32,\n"
            "  fills the padding with zeros, then divides by 32 (the PADDED size, not the\n"
            "  original size). The result is wrong by a factor of (padded_size / original).\n"
            "  This fails silently — no error, just a wrong mean. Same for var.\n\n"
            "✓ CORRECT PATTERN — use ttnn.sum + explicit divide.\n"
            "  ttnn.sum IS correct under padding because pad values are 0 and contribute\n"
            "  0 to the sum. Then divide by the REAL (un-padded) dim size yourself.\n\n"
            "PRIMARY RECIPE — channels-first LayerNorm with sub-tile C:\n"
            "```python\n"
            "import ttnn\n"
            "\n"
            "def channels_first_layer_norm_sub_tile(x_nchw, gamma, beta, eps, C):\n"
            "    '''LayerNorm over the channel dim of NCHW when C < tile_width.\n"
            "    gamma/beta shapes are (1, C, 1, 1) so they broadcast over spatial dims.\n"
            "    '''\n"
            "    N = int(x_nchw.shape[0])\n"
            "    H = int(x_nchw.shape[2])\n"
            "    W = int(x_nchw.shape[3])\n"
            "    inv_c = 1.0 / float(C)\n"
            "\n"
            "    # Move C to last axis so the reduction dim is the last one.\n"
            "    x = ttnn.reshape(x_nchw, (N, C, H * W))   # (N, C, HW)\n"
            "    x = ttnn.transpose(x, -2, -1)              # (N, HW, C)\n"
            "\n"
            "    # mean = sum / C — DO NOT use ttnn.mean here.\n"
            "    sum_x = ttnn.sum(x, dim=-1, keepdim=True)\n"
            "    mean  = ttnn.multiply(sum_x, inv_c)\n"
            "\n"
            "    centered = ttnn.subtract(x, mean)\n"
            "    sq       = ttnn.multiply(centered, centered)\n"
            "    sum_sq   = ttnn.sum(sq, dim=-1, keepdim=True)\n"
            "    var      = ttnn.multiply(sum_sq, inv_c)\n"
            "\n"
            "    var_eps  = ttnn.add(var, eps)\n"
            "    rstd     = ttnn.rsqrt(var_eps)\n"
            "    normed   = ttnn.multiply(centered, rstd)\n"
            "\n"
            "    # Restore layout.\n"
            "    normed = ttnn.transpose(normed, -2, -1)    # (N, C, HW)\n"
            "    normed = ttnn.reshape(normed, (N, C, H, W))\n"
            "\n"
            "    # Affine (gamma/beta have shape (1, C, 1, 1)).\n"
            "    normed = ttnn.multiply(normed, gamma)\n"
            "    normed = ttnn.add(normed, beta)\n"
            "    return normed\n"
            "```\n\n"
            "FALLBACK RECIPE — tile-replicate to fill tile_width (only if PRIMARY blocked):\n"
            "Replicate the sub-tile dim k = 32/C times so the input fills a full tile.\n"
            "Replication preserves mean and variance exactly:\n"
            "  mean_padded = (k * sum_real) / 32 = sum_real / C ✓\n"
            "Then run native ttnn.layer_norm and slice the first C lanes back.\n"
            "```python\n"
            "k = 32 // C                                   # for C=4, k=8\n"
            "x = ttnn.permute(x_nchw, (0, 2, 3, 1))         # (N, H, W, C)\n"
            "x_tiled = ttnn.concat([x] * k, dim=-1)         # (N, H, W, 32)\n"
            "g_tiled = ttnn.concat([gamma] * k, dim=-1)\n"
            "b_tiled = ttnn.concat([beta]  * k, dim=-1)\n"
            "y_tiled = ttnn.layer_norm(x_tiled, weight=g_tiled, bias=b_tiled, epsilon=eps)\n"
            "y = ttnn.slice(y_tiled, [0,0,0,0], [N, H, W, C])\n"
            "y = ttnn.permute(y, (0, 3, 1, 2))              # (N, C, H, W)\n"
            "```\n\n"
            "FORBIDDEN (silently wrong):\n"
            "  ❌  ttnn.mean(x, dim=-1)              when reduction dim < 32\n"
            "  ❌  ttnn.var(x, dim=-1)               same issue\n"
            "  ❌  ttnn.layer_norm(x, ...)          with sub-tile last dim (rejects loudly)\n"
        ),
    )
)

_register(
    Recipe(
        id="padded_layer_norm",
        title="Last-dim padding for LayerNorm when D >= 32 but D % 32 != 0",
        body=(
            "Constraint: last_dim = {last_dim} is not a multiple of tile_width "
            "({tile_width}). Pad to {tile_width} multiples, layer_norm, then slice back.\n\n"
            "```python\n"
            "import ttnn\n"
            "import torch.nn.functional as F\n"
            "\n"
            "TILE = {tile_width}\n"
            "\n"
            "def _last_dim_padded_layer_norm(x, gamma, beta, eps, device):\n"
            "    d = x.shape[-1]                     # e.g. {last_dim}\n"
            "    pad = (-d) % TILE                   # bytes to add\n"
            "    if pad:\n"
            "        x = F.pad(x, (0, pad))\n"
            "        gamma = F.pad(gamma, (0, pad))  # padding gamma with 0 zeros out the slot\n"
            "        beta  = F.pad(beta,  (0, pad))\n"
            "    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)\n"
            "    g_tt = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)\n"
            "    b_tt = ttnn.from_torch(beta,  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)\n"
            "    y_tt = ttnn.layer_norm(x_tt, weight=g_tt, bias=b_tt, epsilon=eps)\n"
            "    if pad:\n"
            "        y_tt = ttnn.slice(y_tt, [0]*len(y_tt.shape),\n"
            "                          list(y_tt.shape[:-1]) + [d])\n"
            "    return y_tt\n"
            "```\n"
        ),
    )
)

_register(
    Recipe(
        id="bias_dtype_match",
        title="Conv/Linear bias must match input dtype",
        body=(
            "Constraint: when the test harness casts the captured input to bf16 but "
            "the HF reference's Conv2d/Linear bias stays fp32, torch raises:\n"
            "  `RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same`\n\n"
            "Two correct fixes — pick ONE, do not combine:\n\n"
            "  Fix A — keep the model on its native dtype and capture inputs to match.\n"
            "          Build the HF reference module without `.to(bfloat16)` and let\n"
            "          the captured inputs stay fp32. The ttnn port can convert internally.\n\n"
            "  Fix B — coerce the bias to the input dtype at module-init time, then\n"
            "          run the HF reference forward on bf16:\n\n"
            "```python\n"
            "class {component}TT:\n"
            "    def __init__(self, hf_module, device):\n"
            "        self._torch_module = hf_module\n"
            "        # Make HF reference happy — promote bias to match expected input dtype.\n"
            "        # The TTNN port doesn't use the torch reference; this only matters\n"
            "        # for the PCC test's reference call.\n"
            "        for sub in hf_module.modules():\n"
            "            if hasattr(sub, 'bias') and sub.bias is not None and sub.bias.dtype != getattr(self, 'input_dtype', sub.bias.dtype):\n"
            "                sub.bias.data = sub.bias.data.to(self.input_dtype)\n"
            "        # ... extract weights, prep ttnn buffers, etc.\n"
            "```\n\n"
            "Detected: captured input dtype is `{input_dtype}` for component `{component}`.\n"
        ),
    )
)

_register(
    Recipe(
        id="position_embedding_shape_dedup",
        title="Position-embedding `shape` arg must not be duplicated",
        body=(
            "The test harness may pass `shape` both positionally (derived from the "
            "captured tensor) AND as a kwarg, triggering:\n"
            "  `TypeError: got multiple values for argument 'shape'`\n\n"
            "In your stub, accept variadic args and pull `shape` from kwargs first, "
            "ignoring any positional override:\n\n"
            "```python\n"
            "class {component}TT:\n"
            "    def __call__(self, *args, **kwargs):\n"
            "        # The harness may pass shape both ways; honor the kwarg.\n"
            "        shape = kwargs.pop('shape', None)\n"
            "        if shape is None and args:\n"
            "            # First positional arg is conventionally the tensor;\n"
            "            # shape is derived from it if not given.\n"
            "            shape = tuple(args[0].shape) if hasattr(args[0], 'shape') else None\n"
            "        # ... compute the embedding with `shape` as the sole reference\n"
            "        return self._embed(shape)\n"
            "```\n"
        ),
    )
)

_register(
    Recipe(
        id="padded_matmul",
        title="Pad K (and weight) to tile_width for matmul",
        body=(
            "Constraint: K = {k_dim} is not a multiple of tile_width ({tile_width}). "
            "ttnn.matmul rejects sub-tile K.\n\n"
            "Pad both the input's last dim AND the weight's matching axis to the next "
            "multiple of {tile_width}, run matmul, then either slice the output or rely on "
            "the unchanged N dim:\n\n"
            "```python\n"
            "import ttnn\n"
            "import torch.nn.functional as F\n"
            "\n"
            "TILE = {tile_width}\n"
            "\n"
            "def _padded_matmul(x, w, device):\n"
            "    K = x.shape[-1]                  # e.g. {k_dim}\n"
            "    pad = (-K) % TILE\n"
            "    if pad:\n"
            "        x = F.pad(x, (0, pad))       # last dim of x\n"
            "        w = F.pad(w, (0, 0, 0, pad)) # matching K axis of w; layout: w shape is (K, N)\n"
            "    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)\n"
            "    w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)\n"
            "    return ttnn.matmul(x_tt, w_tt)\n"
            "```\n\n"
            "Caveat: padding the input with zeros and the weight with zeros leaves the\n"
            "matmul result unchanged for the non-padded entries (0*anything=0). No mask\n"
            "needed.\n"
        ),
    )
)
