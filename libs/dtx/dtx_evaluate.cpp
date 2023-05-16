#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

vector<float> evaluate(vector<float> data, std::vector<uint32_t> address_map, vector<int> output_shape) {
    auto data_transformed_size = vector_product(output_shape);
    vector<float> data_transformed(data_transformed_size, 0);
    for(uint32_t i = 0; i < address_map.size(); i+=4) {
        auto src_address = address_map[i];
        auto dst_address = address_map[i+1];
        auto transfer_size = address_map[i+2];
        auto pad = address_map[i+3];
        for (uint32_t s = 0; s < transfer_size; s++) {
            assert(dst_address+s < data_transformed.size());
            if (pad) {
                data_transformed[dst_address+s] = 0;
            }
            else {
                assert(src_address+s < data.size());
                data_transformed[dst_address+s] = data[src_address+s];
            }
        }
    }
    return data_transformed;
}
