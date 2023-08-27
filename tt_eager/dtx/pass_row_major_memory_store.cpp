#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

using namespace std;

bool row_major_memory_store(DataTransformations * dtx) {
    bool DEBUG = false;

    if (DEBUG) cout << "\n\nPASS: Row Major Memory Store" << endl;

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("row_major_memory_store", producer->groups.size());
    dtx->transformations.push_back(consumer);

    for (int group_idx=0; group_idx<producer->groups.size(); group_idx++) {
        TensorPairGroup * consumer_group = consumer->groups[group_idx];
        TensorPairGroup * producer_group = producer->groups[group_idx];
        inherit_group_attributes_from_producer(producer_group, consumer_group);
        // Calculate producer and consumer shapes
        vector<int> producer_shape = producer_group->shape;
        int rank = producer_shape.size();
        assert(rank == 3); // TODO: generalize for rank
        vector<int> consumer_shape = {1, 1, vector_product(producer_shape)};
        consumer_group->shape = consumer_shape;
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
                                                group_idx,
                                                new DTXTensor({consumer_str}, {consumer_end}));
                if (DEBUG) cout << s(6) << i << ".  " << tp->get_string() << endl;

                consumer_group->tensor_pairs.push_back(tp);

                consumer_x += producer_shape[X(rank)];
                i++;
            }
        }
    }

    return true;
}
