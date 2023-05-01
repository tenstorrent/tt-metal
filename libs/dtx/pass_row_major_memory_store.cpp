#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"


bool row_major_memory_store(DataTransformations * dtx) {
    bool DEBUG = false;

    if (DEBUG) cout << "\n\nPASS: Row Major Memory Store" << endl;

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("row_major_memory_store", producer->groups.size());  // TODO: generalize for groups>1
    dtx->transformations.push_back(consumer);

    // Calculate producer and consumer shapes
    vector<int> producer_shape = producer->groups[0]->shape;
    int rank = producer_shape.size();
    vector<int> consumer_shape = {1, 1, vector_product(producer_shape)};
    consumer->groups[0]->shape = consumer_shape;
    if(DEBUG) cout << "Producer shape - " << v2s(producer_shape) << endl;
    if(DEBUG) cout << "Consumer shape - " << v2s(consumer_shape) << endl;
    // Generate tensor pairs
    int consumer_x = 0;
    int i = 0;
    for (int z=0; z<producer_shape[Z(rank)]; z++) {
        for (int y=0; y<producer_shape[Y(rank)]; y++) {
            vector<int> producer_str = {z, y, 0};
            vector<int> producer_end = {z, y, producer_shape[X(rank)]-1};
            vector<int> consumer_str = {0, 0, consumer_x};
            vector<int> consumer_end = {0, 0, consumer_x + producer_shape[X(rank)]-1};

            TensorPair * tp = new TensorPair(new DTXTensor({producer_str}, {producer_end}),
                                            0,
                                            new DTXTensor({consumer_str}, {consumer_end}));
            if (DEBUG) cout << s(6) << i << ".  " << tp->get_string() << endl;

            consumer->groups[0]->tensor_pairs.push_back(tp);

            consumer_x += producer_shape[X(rank)];
            i++;
        }
    }


    return true;
}
