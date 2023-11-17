import numpy as np

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.untilize_with_halo_config_generation_and_validation import (
    NEIGHBORHOOD_DIST,
    generate_untilize_with_halo_kernel_configs,
)

if __name__ == "__main__":
    ## Inputs:  1. tensor_metadata:             [(is_pad, core_id, local_idx), ...], size = padded tensor size
    ##                                              NOTE: (core_id, local_idx) == global_idx
    ##          2. resharded_start_and_end:     [(core_id, req_shard_start, req_shard_end), ...], size = num cores

    ## some dummy inputs:
    tensor_metadata = [
        ## core 0
        (True, -1, 0),
        (True, -1, 0),
        (True, -1, 0),
        (True, -1, 0),
        (True, -1, 0),
        (False, 0, 0),
        (False, 0, 1),
        (True, -1, 0),
        (True, -1, 0),
        (False, 0, 2),
        (False, 0, 3),
        (True, -1, 0),
        ## core 1
        (True, -1, 0),
        (False, 1, 0),
        (False, 1, 1),
        (True, -1, 0),
        (True, -1, 0),
        (False, 1, 2),
        (False, 1, 3),
        (True, -1, 0),
        (True, -1, 0),
        (True, -1, 0),
        (True, -1, 0),
        (True, -1, 0),
    ]
    resharding = [((0, 0), (0, 15)), ((0, 0), (8, 23))]  ## global idx (inclusive)

    ## corresponding reference output
    ref_local_data = [[(5, 2), (9, 2)], [(5, 2), (9, 2)]]
    ref_local_pad = [[(0, 5), (7, 2), (11, 2), (15, 1)], [(0, 1), (3, 2), (7, 2), (11, 5)]]
    ref_ll_data = [[], []]
    ref_l_data = [[], [(13, 2)]]
    ref_r_data = [[(1, 2)], []]
    ref_rr_data = [[], []]
    ref_src_start_idx = [[-1, -1, 0, 2, -1], [-1, 0, 0, -1, -1]]

    local_data, local_pad, ll_data, l_data, r_data, rr_data, src_start_idx = generate_untilize_with_halo_kernel_configs(
        tensor_metadata, resharding
    )

    print(f"ref local data: {ref_local_data}")
    print(f"local data:     {local_data}\n")
    print(f"ref local pad:  {ref_local_pad}")
    print(f"local pad:      {local_pad}\n")
    print(f"ref ll data:    {ref_ll_data}")
    print(f"ll data:        {ll_data}\n")
    print(f"ref l data:     {ref_l_data}")
    print(f"l data:         {l_data}\n")
    print(f"ref r data:     {ref_r_data}")
    print(f"r data:         {r_data}\n")
    print(f"ref rr data:    {ref_rr_data}")
    print(f"rr data:        {rr_data}\n")
    print(f"ref_src_start_idx:  {ref_src_start_idx}")
    print(f"src_start_idx:      {src_start_idx}\n")
