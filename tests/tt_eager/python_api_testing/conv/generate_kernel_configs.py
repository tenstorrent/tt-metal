import numpy as np

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc

NEIGHBORHOOD_DIST = 2  ## ll, l, r, rr


## Inputs:  1. tensor_metadata:             [(is_pad, src_core_id, src_local_idx), ...], size = padded tensor size
##                                              NOTE: (src_core_id, src_local_idx) == src_global_idx
##          2. resharded_start_and_end:     [(req_shard_start, req_shard_end), ...], size = num cores
##
## Outputs: 1. local_data_start_and_size:   [[(dst_start, size), ...], ...], size = num cores
##          2. local_pad_start_and_size:    [[(dst_start, size), ...], ...], size = num cores
##          3. neighbor data config:            NOTE: currently NEIGHBORHOOD_DIST = 2. Can be generalized.
##              1. ll_send_start_and_size:  [[(dst_start, size), ...], ...], size = num cores
##              2. l_send_start_and_size:   [[(dst_start, size), ...], ...], size = num cores
##              3. r_send_start_and_size:   [[(dst_start, size), ...], ...], size = num cores
##              4. rr_send_start_and_size:  [[(dst_start, size), ...], ...], size = num cores
def generate_kernel_configs(tensor_metadata: list, resharded_start_and_end: list):
    ncores = len(resharded_start_and_end)

    ## data :: { core -> [
    ##              [],    ## ll
    ##              [],    ## l
    ##              [],    ## local
    ##              [],    ## r
    ##              [],    ## rr
    ##          ]}
    core_neighbor_data = {}
    core_pad_start_and_size = {}
    core_src_local_start_idx = {}  ## {core -> [ ll, l, local, r, rr ]}

    ## NOTE: assuming the core_id's are contiguous
    for dst_core_id in np.arange(ncores):
        ## generate the config for dst_core_id using the input metadata

        dst_global_start_idx, dst_global_end_idx = resharded_start_and_end[dst_core_id]

        curr_segment_size = 0
        is_curr_segment_pad = None
        curr_segment_src_core_id = None
        curr_segment_dst_start_idx = None
        curr_segment_neighbor_idx = None

        for dst_global_idx in np.arange(dst_global_start_idx, dst_global_end_idx + 1):
            dst_local_idx = dst_global_idx - dst_global_start_idx
            is_pad, src_core_id, src_local_idx = tensor_metadata[dst_global_idx]

            if is_pad:  ## pad stick
                if curr_segment_size > 0 and is_curr_segment_pad:
                    ## current segment is padding
                    curr_segment_size += 1
                else:
                    if curr_segment_size > 0:
                        ## current segment is data, a new pad segment starts here
                        ## finish off the data seg first
                        if curr_segment_src_core_id not in core_neighbor_data:
                            core_neighbor_data[curr_segment_src_core_id] = []
                            for i in np.arange(2 * NEIGHBORHOOD_DIST + 1):
                                core_neighbor_data[curr_segment_src_core_id].append([])
                        core_neighbor_data[curr_segment_src_core_id][curr_segment_neighbor_idx].append(
                            (curr_segment_dst_start_idx, curr_segment_size)
                        )
                    else:
                        ## there is no current segment
                        pass
                    ## start new pad segment
                    is_curr_segment_pad = True
                    curr_segment_size = 1
                    curr_segment_dst_start_idx = dst_local_idx

            else:  ## data stick
                ## the neighbor core of dst_core_id this data stick is coming from (src_core_id): ll, l, local, r or rr
                neighbor_idx = NEIGHBORHOOD_DIST + (dst_core_id - src_core_id)

                if curr_segment_size > 0:
                    if curr_segment_src_core_id == src_core_id:
                        ## this data stick belong to the same src core as current segment
                        ## if the curr segment is also data, then it is contiguous
                        ## else, this is new data segment after a pad break
                        if not is_curr_segment_pad:
                            ## contiguous data stick
                            curr_segment_size += 1
                        else:
                            ## curr segment is padding, and a new data segment starts here
                            ## finish off the pad segment first (always local only)
                            if dst_core_id not in core_pad_start_and_size:
                                core_pad_start_and_size[dst_core_id] = []
                            core_pad_start_and_size[dst_core_id].append((curr_segment_dst_start_idx, curr_segment_size))
                            ## start the new data segment
                            is_curr_segment_pad = False
                            curr_segment_size = 1
                            curr_segment_dst_start_idx = dst_local_idx
                            curr_segment_src_core_id = src_core_id
                            curr_segment_neighbor_idx = neighbor_idx
                    else:
                        if not is_curr_segment_pad:
                            ## this data stick belongs to a different src core than the current data segment
                            ## first finish the current data segment
                            if curr_segment_src_core_id not in core_neighbor_data:
                                core_neighbor_data[curr_segment_src_core_id] = []
                                for i in np.arange(2 * NEIGHBORHOOD_DIST + 1):
                                    core_neighbor_data[curr_segment_src_core_id].append([])
                            core_neighbor_data[curr_segment_src_core_id][curr_segment_neighbor_idx].append(
                                (curr_segment_dst_start_idx, curr_segment_size)
                            )
                        else:
                            ## current segment is padding, finish it off
                            if dst_core_id not in core_pad_start_and_size:
                                core_pad_start_and_size[dst_core_id] = []
                            core_pad_start_and_size[dst_core_id].append((curr_segment_dst_start_idx, curr_segment_size))
                        ## start the new data segment
                        is_curr_segment_pad = False
                        curr_segment_size = 1
                        curr_segment_dst_start_idx = dst_local_idx
                        curr_segment_src_core_id = src_core_id
                        curr_segment_neighbor_idx = neighbor_idx
                else:
                    ## there is no current segment, create new data segment
                    is_curr_segment_pad = False
                    curr_segment_size = 1
                    curr_segment_dst_start_idx = dst_local_idx
                    curr_segment_src_core_id = src_core_id
                    curr_segment_neighbor_idx = neighbor_idx

        ## finish off the remaining last segment, if any
        if curr_segment_size > 0:
            if is_curr_segment_pad:
                ## padding segment
                if dst_core_id not in core_pad_start_and_size:
                    core_pad_start_and_size[dst_core_id] = []
                core_pad_start_and_size[dst_core_id].append((curr_segment_dst_start_idx, curr_segment_size))
            else:
                ## data segment
                if curr_segment_src_core_id not in core_neighbor_data:
                    core_neighbor_data[curr_segment_src_core_id] = []
                    for i in np.arange(2 * NEIGHBORHOOD_DIST + 1):
                        core_neighbor_data[curr_segment_src_core_id].append([])
                core_neighbor_data[curr_segment_src_core_id][curr_segment_neighbor_idx].append(
                    (curr_segment_dst_start_idx, curr_segment_size)
                )

    # print(tensor_metadata)
    # print(resharded_start_and_end)

    print(core_neighbor_data)
    # print(core_pad_start_and_size)

    ll_data_start_and_size = []
    l_data_start_and_size = []
    local_data_start_and_size = []
    r_data_start_and_size = []
    rr_data_start_and_size = []
    local_pad_start_and_size = []
    for i in range(ncores):
        ll_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST - 2])
        l_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST - 1])
        local_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST])
        r_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST + 1])
        rr_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST + 2])
        local_pad_start_and_size.append(core_pad_start_and_size[i])

    return (
        local_data_start_and_size,
        local_pad_start_and_size,
        ll_data_start_and_size,
        l_data_start_and_size,
        r_data_start_and_size,
        rr_data_start_and_size,
    )


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
resharding = [(0, 15), (8, 23)]  ## global idx (inclusive)

## corresponding reference output
ref_local_data = [[(5, 2), (9, 2)], [(5, 2), (9, 2)]]
ref_local_pad = [[(0, 5), (7, 2), (11, 2), (15, 1)], [(0, 1), (3, 2), (7, 2), (11, 5)]]
ref_ll_data = [[], []]
ref_l_data = [[], [(13, 2)]]
ref_r_data = [[(1, 2)], []]
ref_rr_data = [[], []]

local_data, local_pad, ll_data, l_data, r_data, rr_data = generate_kernel_configs(tensor_metadata, resharding)

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
