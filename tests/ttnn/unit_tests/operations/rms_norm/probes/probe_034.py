import ttnn, torch
from ttnn.operations.rms_norm.rms_norm import create_program_descriptor as cpd

pd = cpd.__globals__
device = ttnn.open_device(device_id=0)
try:
    print("AICLK MHz:", device.get_clock_rate_mhz() if hasattr(device, "get_clock_rate_mhz") else "n/a")
    print("grid:", device.compute_with_storage_grid_size())
    print("l1_size_per_core bound:", hasattr(device, "l1_size_per_core"))
    print("max_worker_l1_unreserved:", ttnn.get_max_worker_l1_unreserved_size())
    for W in (1024, 2304, 5120, 7168):
        shape = (1, 1, 8192, W)
        x = ttnn.from_torch(
            torch.zeros(1, 1, 32, W, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # fake a big tensor cheaply: just call _plan with a real tensor of the target shape
        del x
        t = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        bytes_ = {"in_tile": 2048, "out_tile": 2048, "gamma_tile": 2048, "stat_tile": 4096, "bf16_tile": 2048}
        p = pd["_plan"](device, t, has_gamma=True, bytes_=bytes_)
        print(
            W,
            {
                k: p[k]
                for k in (
                    "num_row_groups",
                    "num_hidden_slices",
                    "slice_hidden_tiles",
                    "block_rows",
                    "rect_w",
                    "rect_h",
                    "row_tiles",
                    "hidden_tiles",
                )
            },
        )
        fb = pd["_footprint_bytes"](
            p["block_rows"],
            p["slice_hidden_tiles"],
            p["num_hidden_slices"],
            is_row_major=False,
            has_gamma=True,
            bytes_=bytes_,
        )
        print("   footprint KB:", fb / 1024, "budget KB:", pd["_l1_working_budget"](device) / 1024)
        del t
finally:
    ttnn.close_device(device)
