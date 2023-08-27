#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

using namespace std;

bool pad_2d_matrix(DataTransformations * dtx, vector<int> pad_to_nearest) {
    bool DEBUG = false;
    if(DEBUG) cout << "Pad 2d matrix " << endl;
    assert(pad_to_nearest.size() == 2);
    assert(dtx->transformations.size() > 0);
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("pad_2d_matrix", producer->groups.size());
    dtx->transformations.push_back(consumer);
    for(auto g_idx = 0; g_idx < producer->groups.size(); g_idx++) {
        TensorPairGroup * consumer_group = consumer->groups[g_idx];
        TensorPairGroup * producer_group = producer->groups[g_idx];
        inherit_group_attributes_from_producer(producer_group, consumer_group);
        vector<int> producer_shape = producer_group->shape;
        int rank = producer_group->shape.size();
        assert(rank >= 2 && rank <= 4);
        auto consumer_shape = producer_shape;
        consumer_shape[X(rank)] = std::ceil((double) producer_shape[X(rank)] / (double) pad_to_nearest[1]) * pad_to_nearest[1];
        consumer_shape[Y(rank)] = std::ceil((double) producer_shape[Y(rank)] / (double) pad_to_nearest[0]) * pad_to_nearest[0];
        consumer_group->shape = consumer_shape;
        if(DEBUG) cout << "Producer shape - " << v2s(producer_shape) << endl;
        if(DEBUG) cout << "Consumer shape - " << v2s(consumer_shape) << endl;
        int W_ = W(rank) >= 0 ? producer_shape[W(rank)] : 1;
        int Z_ = Z(rank) >= 0 ? producer_shape[Z(rank)] : 1;
        int i = 0;
        vector<int> str(rank, 0);
        for(auto w = 0; w < W_; w++) {
            if(W(rank) >= 0) {
                str[W(rank)] = w;
            }
            for(auto z = 0; z < Z_; z++) {
                if(Z(rank) >= 0) {
                    str[Z(rank)] = z;
                }
                vector<int> end(str);
                for(int y = 0; y < producer_shape[Y(rank)]; y++) {
                    str[Y(rank)] = y;
                    str[X(rank)] = 0;
                    end[Y(rank)] = y;
                    end[X(rank)] = producer_shape[X(rank)]-1;
                    vector<int> consumer_str(str);
                    vector<int> consumer_end(end);
                    consumer_group->tensor_pairs.push_back(new TensorPair(new DTXTensor(str, end),
                                                    g_idx,
                                                    new DTXTensor(consumer_str, consumer_end)));
                    i++;
                    if (DEBUG) cout << s(6) << i << ".  " << consumer_group->tensor_pairs.back()->get_string() << endl;
                    auto pad_width = consumer_shape[X(rank)] - producer_shape[X(rank)];
                    if(pad_width > 0) {
                        // Pad x-dim. Partial row of 0.
                        consumer_str[X(rank)] = producer_shape[X(rank)];
                        consumer_end[X(rank)] = consumer_shape[X(rank)] - 1;
                        str[Y(rank)] = 0;
                        str[X(rank)] = 0;
                        end[Y(rank)] = 0;
                        end[X(rank)] = pad_width - 1;
                        consumer_group->tensor_pairs.push_back(new TensorPair(new DTXTensor(str, end),
                                                    -1,
                                                    new DTXTensor(consumer_str, consumer_end)));
                        i++;
                        if (DEBUG) cout << s(6) << i << ".  " << consumer_group->tensor_pairs.back()->get_string() << endl;
                    }
                }
                auto pad_height = consumer_shape[Y(rank)] - producer_shape[Y(rank)];
                if(pad_height > 0) {
                    // Pad y-dim with rows of zeroes
                    for (int ph = 0; ph < pad_height; ph++) {
                        auto pad_width = consumer_shape[X(rank)];
                        str[Y(rank)] = 0;
                        str[X(rank)] = 0;
                        end[Y(rank)] = 0;
                        end[X(rank)] = pad_width - 1;
                        vector<int> consumer_str(str);
                        consumer_str[Y(rank)] = producer_shape[Y(rank)] + ph;
                        consumer_str[X(rank)] = 0;
                        vector<int> consumer_end(consumer_str);
                        consumer_end[X(rank)] = pad_width - 1;
                        consumer_group->tensor_pairs.push_back(new TensorPair(new DTXTensor(str, end),
                                                        -1,
                                                        new DTXTensor(consumer_str, consumer_end)));
                        i++;
                        if (DEBUG) cout << s(6) << i << ".  " << consumer_group->tensor_pairs.back()->get_string() << endl;
                    }
                }
            }
        }
    }
    //exit(1);
    return true;
}
