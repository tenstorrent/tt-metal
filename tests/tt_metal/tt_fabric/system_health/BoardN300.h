#pragma once

#include <string>
#include <unordered_map>
#include "BoardType.h"
#include <iostream>
class N300Board : public BoardType {
protected:
    const int32_t num_asics = 2;
    const int32_t num_ports = 2;
    int32_t num_channels_per_asic = 16;

    void initialize_connection_maps() override {
        conn_to_endpoint_map.clear();
        endpoint_to_conn_map.clear();

        conn_to_endpoint_map[(connection_u){QSFP, 1}.as_uint64] = {1, {6, 7}};
        conn_to_endpoint_map[(connection_u){QSFP, 2}.as_uint64] = {1, {0, 1}};
        conn_to_endpoint_map[(connection_u){WARP, 1}.as_uint64] = {1, {14, 15}};
        conn_to_endpoint_map[(connection_u){WARP, 2}.as_uint64] = {2, {6, 7}};
        // TODO: ARBITRARY IDs for INTERNAL
        conn_to_endpoint_map[(connection_u){INTERNAL, 1}.as_uint64] = {1, {8, 9}};
        conn_to_endpoint_map[(connection_u){INTERNAL, 2}.as_uint64] = {2, {0, 1}};

        endpoint_to_conn_map[(asic_chan_u){1, 6}.as_uint64] = {QSFP, 1};
        endpoint_to_conn_map[(asic_chan_u){1, 7}.as_uint64] = {QSFP, 1};
        endpoint_to_conn_map[(asic_chan_u){1, 0}.as_uint64] = {QSFP, 2};
        endpoint_to_conn_map[(asic_chan_u){1, 1}.as_uint64] = {QSFP, 2};
        endpoint_to_conn_map[(asic_chan_u){1, 14}.as_uint64] = {WARP, 1};
        endpoint_to_conn_map[(asic_chan_u){1, 15}.as_uint64] = {WARP, 1};
        endpoint_to_conn_map[(asic_chan_u){2, 6}.as_uint64] = {WARP, 2};
        endpoint_to_conn_map[(asic_chan_u){2, 7}.as_uint64] = {WARP, 2};
        endpoint_to_conn_map[(asic_chan_u){1, 8}.as_uint64] = {INTERNAL, 1};
        endpoint_to_conn_map[(asic_chan_u){1, 9}.as_uint64] = {INTERNAL, 1};
        endpoint_to_conn_map[(asic_chan_u){2, 0}.as_uint64] = {INTERNAL, 2};
        endpoint_to_conn_map[(asic_chan_u){2, 1}.as_uint64] = {INTERNAL, 2};
    };

public:
    N300Board() : BoardType() {
        // Should ensure that maps are initialized
        initialize_connection_maps();
    }

    virtual ~N300Board() = default;

    // Returns the number of chips on the board
    int get_chip_count() const { return num_asics; };
};
