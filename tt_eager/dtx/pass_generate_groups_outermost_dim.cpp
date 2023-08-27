#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

using namespace std;

bool generate_groups_outermost_dim(DataTransformations * dtx) {
    bool DEBUG = false;

    if (DEBUG) cout << "\n\nPASS: Generate groups on outermost dimension" << endl;

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    assert(producer->groups.size() == 1);
    TensorPairGroup * producer_group = producer->groups[0];
    auto producer_shape = producer_group->shape;
    uint rank = producer_shape.size();
    assert(rank == 3); // TODO: generalize for rank != 3
    uint32_t num_consumer_groups = producer_shape[0];
    if (DEBUG) std::cout << "Number of consumer groups - " << num_consumer_groups << std::endl;
    TransformationNode * consumer = new TransformationNode("generate_groups", num_consumer_groups);
    dtx->transformations.push_back(consumer);
    for (int g=0; g<num_consumer_groups; g++) {
        TensorPairGroup * consumer_group = consumer->groups[g];
        consumer_group->shape = {1, producer_shape[1], producer_shape[2]};
        vector<int> producer_str = {g, 0, 0};
        vector<int> producer_end = {g, producer_shape[1]-1, producer_shape[2]-1};
        vector<int> consumer_str = {0, 0, 0};
        vector<int> consumer_end = {0, producer_shape[1]-1, producer_shape[2]-1};

        TensorPair * tp = new TensorPair(new DTXTensor({producer_str}, {producer_end}),
                                        0,
                                        new DTXTensor({consumer_str}, {consumer_end}));
        if (DEBUG) cout << s(6) << g << ".  " << tp->get_string() << endl;

        consumer_group->tensor_pairs.push_back(tp);
    }

    return true;
}
