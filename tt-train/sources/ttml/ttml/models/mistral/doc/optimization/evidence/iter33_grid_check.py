"""Open the TP=4 mesh, print the compute grid, time 4-layer training steps (random init) at TP=4."""
import sys, time
import ttnn, ttml

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
g = dev.compute_with_storage_grid_size()
print(f"compute grid {g.x} x {g.y} = {g.x * g.y} cores, devices {dev.get_num_devices()}", flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
