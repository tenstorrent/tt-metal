import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout

# --- Device setup ---
device = ttnn.CreateDevice(0)
torch.manual_seed(0)


# Small helper to move back/forth the same way every time
def to_ttnn(x):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)


def to_torch(x):
    return ttnn.to_torch(x)


# Use a size that isn't a multiple of 32 to exercise TTNN tile padding paths
x = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (1, 3, 17, 19), dtype=torch.int32)
x_ttnn = to_ttnn(x)

# ========== MAX (reduction) ==========
# Note: to match TTNN reduction semantics (values only), use torch.amax instead of torch.max(dim)[0]
# A) over all dims -> scalar
torch_max_all = torch.amax(x)
ttnn_max_all = ttnn.max(x_ttnn)  # no dim => over all dims
ttnn_max_all_torch = to_torch(ttnn_max_all)
assert_with_pcc(torch_max_all, ttnn_max_all_torch)
print("max(all): PASS")

# B) over a specific dim
dim = 3
keepdim = True
torch_max_dim = torch.amax(x, dim=dim, keepdim=keepdim)
ttnn_max_dim = ttnn.max(x_ttnn, dim=dim, keepdim=keepdim)
ttnn_max_dim_torch = to_torch(ttnn_max_dim)
assert_with_pcc(torch_max_dim, ttnn_max_dim_torch)
print(f"max(dim={dim}, keepdim={keepdim}): PASS")

# ========== MIN (reduction) ==========
# A) over all dims -> scalar
torch_min_all = torch.amin(x)
ttnn_min_all = ttnn.min(x_ttnn)  # no dim => over all dims
ttnn_min_all_torch = to_torch(ttnn_min_all)
assert_with_pcc(torch_min_all, ttnn_min_all_torch)
print("min(all): PASS")

# B) over a specific dim
dim = 1
keepdim = False
torch_min_dim = torch.amin(x, dim=dim, keepdim=keepdim)
ttnn_min_dim = ttnn.min(x_ttnn, dim=dim, keepdim=keepdim)
ttnn_min_dim_torch = to_torch(ttnn_min_dim)
assert_with_pcc(torch_min_dim, ttnn_min_dim_torch)
print(f"min(dim={dim}, keepdim={keepdim}): PASS")
