from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc

NEIGHBORHOOD_DIST = 2  ## ll, l, r, rr


## Inputs:  1. tensor_metadata:             [(is_pad, core_id, local_idx), ...], size = padded tensor size
##                                              NOTE: (core_id, local_idx) == global_idx
##          2. resharded_start_and_end:     [(core_id, req_shard_start, req_shard_end), ...], size = num cores
##
## Outputs: 1. local_data_start_and_size:   [[(start, size), ...], ...], size = num cores
##          2. local_pad_start_and_size:    [[(start, size), ...], ...], size = num cores
##          3. neighbor data config:            NOTE: currently NEIGHBORHOOD_DIST = 2. Can be generalized.
##              1. ll_send_start_and_size:  [[(start, size), ...], ...], size = num cores
##              2. l_send_start_and_size:   [[(start, size), ...], ...], size = num cores
##              3. r_send_start_and_size:   [[(start, size), ...], ...], size = num cores
##              4. rr_send_start_and_size:  [[(start, size), ...], ...], size = num cores
def generate_kernel_configs(tensor_metadata: list, resharded_start_and_end: list):
    ncores = len(resharded_start_and_end)

    ## data :: { core -> {
    ##              0 -> [],    ## ll
    ##              1 -> [],    ## l
    ##              2 -> [],    ## local
    ##              3 -> [],    ## r
    ##              4 -> []     ## rr
    ##          }}
    core_neighbor_data = {}
    local_pad_start_and_size = []

    for req_core_id, req_start, req_end in resharded_start_and_end:
        curr_segment_core_id = None
        is_curr_segment_pad = None
        curr_segment_start_idx = None
        curr_segment_size = 0
        curr_segment_neighbor_idx = None

        for is_pad, core_id, local_idx in tensor_metadata[req_start:req_end]:
            ## core_id is the source core, req_core_id is the destination core

            ## check which core this stick belongs to: ll, l, local, r or rr
            ## NOTE: assuming the core_id's are contiguous
            neighbor_idx = NEIGHBORHOOD_DIST + (core_id - req_core_id)

            if is_pad:  ## pad stick
                if curr_segment_core_id == core_id:
                    ## this belongs to the same core as current segment
                    if is_curr_segment_pad:
                        ## if prev stick was also pad, count it
                        curr_segment_size += 1
                    else:
                        ## curr segment is data, a new pad segment starts
                        ## finish off the data seg first
                        core_neighbor_data[curr_segment_core_id][curr_segment_neighbor_idx].append(
                            (curr_segment_start_idx, curr_segment_size)
                        )
                        ## start new pad segment
                        is_curr_segment_pad = True
                        curr_segment_size = 1
                        curr_segment_start_idx = local_idx
                        curr_segment_core_id = core_id
                        curr_segment_neighbor_idx = neighbor_idx
                else:
                    ## this stick belongs to a different core than the current segment
                    ## finish off the current segment first
                    if is_curr_segment_pad:
                        if curr_segment_neighbor_idx == NEIGHBORHOOD_DIST:
                            ## curr segment is local pad, save it
                            local_pad_start_and_size.append((curr_segment_start_idx, curr_segment_size))
                        else:
                            ## this is neighbor padding, throw it out
                            pass
                    else:
                        ## curr segment is data
                        ## finish it off first
                        if curr_segment_core_id not in core_neighbor_data:
                            core_neighbor_data[curr_segment_core_id] = {}
                            for i in range(2 * NEIGHBORHOOD_DIST + 1):
                                core_neighbor_data[curr_segment_core_id][i] = []
                        core_neighbor_data[curr_segment_core_id][curr_segment_neighbor_idx].append(
                            (curr_segment_start_idx, curr_segment_size)
                        )

                    ## start a new pad segment
                    is_curr_segment_pad = True
                    curr_segment_size = 1
                    curr_segment_start_idx = local_idx
                    curr_segment_core_id = core_id
                    curr_segment_neighbor_idx = neighbor_idx

            else:  ## data stick
                if curr_segment_core_id == core_id:
                    ## this data stick belong to the same core as current segment
                    ## if the curr segment is also data, then it is contiguous, else this is next data segment after a pad break
                    if not is_curr_segment_pad:
                        ## contiguous data stick
                        curr_segment_size += 1
                    else:
                        ## curr segment is padding, and a new data segment starts
                        ## finish off the pad segment first
                        local_pad_start_and_size.append((curr_segment_start_idx, curr_segment_size))
                        ## start the new data segment
                        is_curr_segment_pad = False
                        curr_segment_size = 1
                        curr_segment_start_idx = local_idx
                        curr_segment_core_id = core_id
                        curr_segment_neighbor_idx = neighbor_idx
                else:
                    ## this data stick belongs to a different core than the current data segment
                    ## first finish the previous segment
                    if curr_segment_core_id not in core_neighbor_data:
                        core_neighbor_data[curr_segment_core_id] = {}
                        for i in range(2 * NEIGHBORHOOD_DIST + 1):
                            core_neighbor_data[curr_segment_core_id][i] = []
                    core_neighbor_data[curr_segment_core_id][curr_segment_neighbor_idx].append(
                        (curr_segment_start_idx, curr_segment_size)
                    )
                    ## start the new data segment
                    is_curr_segment_pad = False
                    curr_segment_size = 1
                    curr_segment_start_idx = local_idx
                    curr_segment_core_id = core_id
                    curr_segment_neighbor_idx = neighbor_idx

    ll_data_start_and_size = core_neighbor_data[:][NEIGHBORHOOD_DIST - 2]
    l_data_start_and_size = core_neighbor_data[:][NEIGHBORHOOD_DIST - 1]
    local_data_start_and_size = core_neighbor_data[:][NEIGHBORHOOD_DIST]
    r_data_start_and_size = core_neighbor_data[:][NEIGHBORHOOD_DIST + 1]
    rr_data_start_and_size = core_neighbor_data[:][NEIGHBORHOOD_DIST + 2]

    return (
        local_data_start_and_size,
        local_pad_start_and_size,
        ll_data_start_and_size,
        l_data_start_and_size,
        r_data_start_and_size,
        rr_data_start_and_size,
    )
