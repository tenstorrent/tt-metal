#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

template <typename T>
vector<T> evaluate(vector<T> data, DataTransformations * dtx) {
    assert(dtx->transformations.size() == 2);
    // copy transfer addresses into a vector
    std::vector<uint32_t> address_map;

    // Generate address map
    for(auto transfer : dtx->transformations.back()->groups[0]->transfers){
        address_map.push_back(transfer->src_address);
        address_map.push_back(transfer->dst_address);
        address_map.push_back(transfer->size);
    }

    auto data_transformed_size = vector_product(dtx->transformations.back()->groups[0]->shape);
    vector<T> data_transformed(data_transformed_size, 0);
    for(uint32_t i = 0; i < address_map.size(); i+=3) {
        auto src_address = address_map[i];
        auto dst_address = address_map[i+1];
        auto transfer_size = address_map[i+2];
        for (uint32_t s = 0; s < transfer_size; s++) {
            assert(dst_address+s < data_transformed.size());
            assert(src_address+s < data.size());
            data_transformed[dst_address+s] = data[src_address+s];
        }
    }
    return data_transformed;
}
