import ttnn
import torch
from models.utility_functions import (
    skip_for_wormhole_b0,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
)

sharded_ln_input = torch.load("torch_sharded_ln_input.pt")
ln_input = torch.load("torch_ln_input.pt")

sharded_ln_output = torch.load("torch_sharded_ln_output.pt")
ln_output = torch.load("torch_ln_output.pt")

sharded_stats = torch.load("torch_sharded_stats.pt")
good_sharded_stats = torch.load("torch_good_sharded_stats.pt")

sharded_after_stats = torch.load("torch_sharded_after_stats.pt")
good_sharded_after_stats = torch.load("torch_good_sharded_after_stats.pt")

pass_value, pcc = comp_pcc(sharded_ln_input, ln_input)
print(pass_value, pcc)

pass_value, pcc = comp_pcc(sharded_ln_output, ln_output)
print(pass_value, pcc)
breakpoint()

pass_value, pcc = comp_pcc(sharded_stats, good_sharded_stats)
print(pass_value, pcc)

pass_value, pcc = comp_pcc(sharded_after_stats, good_sharded_after_stats)
print(pass_value, pcc)
