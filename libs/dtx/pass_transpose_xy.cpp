#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"


bool transpose_xy(DataTransformations * dtx) {
    bool DEBUG = false;

    if (DEBUG) cout << "\n\nPASS: Transpose XY" << endl;

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("transpose_xy", producer->groups.size());  // TODO: generalize for groups>1
    dtx->transformations.push_back(consumer);

    vector<int> producer_shape = producer->groups[0]->shape;
    vector<int> consumer_shape = producer->groups[0]->shape;
    int rank = consumer_shape.size();
    int temp_x = consumer_shape[rank-1];
    consumer_shape[rank-1] = consumer_shape[rank-2];
    consumer_shape[rank-2] = temp_x;

    consumer->groups[0]->shape = consumer_shape;

    return true;
}
