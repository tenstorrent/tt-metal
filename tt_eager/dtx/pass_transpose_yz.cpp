#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

using namespace std;

vector<int> flip_yz_dims(vector<int> input) {
    int rank = input.size();
    vector<int> output = input;
    output[Y(rank)] = input[Z(rank)];
    output[Z(rank)] = input[Y(rank)];
    return output;
}

bool transpose_yz(DataTransformations * dtx) {
    bool DEBUG = false;

    if (DEBUG) cout << "\n\nPASS: Transpose XY" << endl;


    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("transpose_yz", producer->groups.size());  // TODO: generalize for groups>1
    dtx->transformations.push_back(consumer);

    vector<int> producer_shape = producer->groups[0]->shape;
    int rank = producer_shape.size();


    // Calculate the consumer shape
    vector<int> consumer_shape = producer->groups[0]->shape;
    int temp_y = consumer_shape[Y(rank)];
    consumer_shape[Y(rank)] = consumer_shape[Z(rank)];
    consumer_shape[Z(rank)] = temp_y;
    consumer->groups[0]->shape = consumer_shape;

    // Stick to move around
    vector<int> yz_stick_shape = consumer_shape;
    yz_stick_shape[Y(rank)] = 1;
    yz_stick_shape[Z(rank)] = 1;

    vector<int> yz_ones = zeros(rank);
    yz_ones[Y(rank)] = 1;
    yz_ones[Z(rank)] = 1;

    if (DEBUG) cout << s(2) << "yz_stick_shape = " << v2s(yz_stick_shape) << "\n" << endl;

    for (int producer_y=0; producer_y<producer_shape[Y(rank)]; producer_y++) {
        for (int producer_z=0; producer_z<producer_shape[Z(rank)]; producer_z++) {

            // Source start
            vector<int> src_str = zeros(rank); //
            src_str[Y(rank)] = producer_y;
            src_str[Z(rank)] = producer_z;

            // Source end
            vector<int> src_end = vector_subtraction(producer_shape, ones(rank));
            src_end[Y(rank)] = producer_y;
            src_end[Z(rank)] = producer_z;


            // Destination Start
            vector<int> dst_str = zeros(rank);
            dst_str[Y(rank)] = producer_z;
            dst_str[Z(rank)] = producer_y;

            // Destination end
            vector<int> dst_end = vector_subtraction(consumer_shape, ones(rank));
            dst_end[Y(rank)] = producer_z;
            dst_end[Z(rank)] = producer_y;

            TensorPair * tp = new TensorPair(new DTXTensor({src_str}, {src_end}),
                                            0,
                                            new DTXTensor({dst_str}, {dst_end}));
            consumer->groups[0]->tensor_pairs.push_back(tp);

            if (DEBUG) cout << s(2) << "src = " << v2s(src_str) << "-" << v2s(src_end) << " ==> " << v2s(dst_str) << "-" << v2s(dst_end) << endl;
        }
    }

    return true;
}
