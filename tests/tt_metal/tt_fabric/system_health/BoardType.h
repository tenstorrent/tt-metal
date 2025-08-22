#pragma once

#include <string>
#include <unordered_map>
#include <iostream>
typedef enum { INTERNAL, WARP, LINKING_BOARD, QSFP, UNKNOWN } channel_type_t;
typedef uint32_t asic_id;
typedef uint32_t channel_id;
typedef std::pair<channel_type_t, int32_t> connection_t;
typedef std::pair<uint32_t, uint32_t> chan_range_t;
typedef std::pair<asic_id, chan_range_t> asic_chan_range_t;
typedef std::pair<asic_id, uint32_t> asic_chan_t;

class BoardType {
public:
    BoardType() {};

    virtual ~BoardType() = default;

    // Returns the name of the board type
    // virtual std::string get_name() const = 0;

    // Returns the number of chips on the board
    // virtual int get_chip_count() const = 0;

    asic_chan_range_t get_asic_end_from_connection(const connection_t& conn) {
        connection_u key;
        key.type = conn.first;
        key.channel_id = conn.second;
        if (conn_to_endpoint_map.find(key.as_uint64) != conn_to_endpoint_map.end()) {
            return conn_to_endpoint_map[key.as_uint64];
        }
        return {UNKNOWN, {0, 0}};  // Return UNKNOWN if not found
    };
    virtual connection_t get_connection_from_asic_end(const asic_chan_t& end) {
        asic_chan_u key;
        key.asic_id = end.first;
        key.channel_id = end.second;
        if (endpoint_to_conn_map.find(key.as_uint64) != endpoint_to_conn_map.end()) {
            return endpoint_to_conn_map[key.as_uint64];
        }
        return {UNKNOWN, 0};  // Return UNKNOWN if not found
    };

protected:
    // Defining packed structures to be used as keys
    typedef union {
        struct __attribute__((packed)) {
            uint8_t asic_id;
            uint8_t channel_id;
        };
        uint64_t as_uint64;
    } asic_chan_u;

    typedef union {
        struct __attribute__((packed)) {
            channel_type_t type;
            uint32_t channel_id;
        };
        uint64_t as_uint64;
    } connection_u;
    std::unordered_map<uint64_t, asic_chan_range_t> conn_to_endpoint_map;
    std::unordered_map<uint64_t, connection_t> endpoint_to_conn_map;
    virtual void initialize_connection_maps() = 0;

private:
    int32_t num_asics = 0;
    int32_t num_ports = 0;
    int32_t num_channels_per_asic = 0;
};
