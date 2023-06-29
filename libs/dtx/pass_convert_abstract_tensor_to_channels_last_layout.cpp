#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"
#include "dtx_passes.hpp"

using namespace std;

bool convert_abstract_tensor_to_channels_last_layout(DataTransformations * dtx){
    bool DEBUG = false;
    bool pass = true;
    if (DEBUG) tt::log_info(tt::LogDTX, "PASS: convert_abstract_tensor_to_channels_last_layout - START");

    // First add the 2 required transposes
    pass &= transpose_yz(dtx);
    pass &= transpose_xy(dtx);

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("convert_abstract_tensor_to_channels_last_layout", producer->groups.size());
    dtx->transformations.push_back(consumer);

    // Calculate producer shape
    vector<int> producer_shape = producer->groups[0]->shape;
    int rank = producer_shape.size();

    // Calculate the consumer shape
    int consumer_shape_x = vector_product(producer_shape);
    int consumer_shape_y = 1;
    consumer->groups[0]->shape = {1, consumer_shape_y, consumer_shape_x};

    int consumer_y = 0;
    int consumer_x = 0;
    for (int producer_y=0; producer_y<producer_shape[Y(rank)]; producer_y++) {
    for (int producer_z=0; producer_z<producer_shape[Z(rank)]; producer_z++) {

        if (DEBUG) tt::log_info(tt::LogDTX, "{}producer_y/z = {}, {}", s(2), producer_z, producer_y);


        vector<int> sweep_zy = { producer_z, producer_y};

        if (DEBUG) tt::log_info("{}sweep_zy = {}", s(4), "sweep_zy = ", v2s(sweep_zy));

        // Producer str/end
        vector<int> producer_str = { producer_z, producer_y, 0};
        vector<int> producer_end = { producer_z, producer_y, producer_shape[X(rank)]-1};

        vector<int> consumer_str = {0, consumer_y, consumer_x};
        vector<int> consumer_end = {0, consumer_y, consumer_x + producer_shape[X(rank)]-1};

        TensorPair * tp = new TensorPair(new DTXTensor({producer_str}, {producer_end}),
                                        0,
                                        new DTXTensor({consumer_str}, {consumer_end}));
        consumer->groups[0]->tensor_pairs.push_back(tp);

        if (DEBUG) tt::log_info(tt::LogDTX, "{}src = {} - {} ==> {} - {}", s(6), v2s(producer_str), v2s(producer_end), v2s(consumer_str), v2s(consumer_end));

        consumer_x += producer_shape[X(rank)]; // length of channel
    }}

    if (DEBUG) tt::log_info(tt::LogDTX, "PASS: convert_abstract_tensor_to_channels_last_layout - END");
    return true;
}
