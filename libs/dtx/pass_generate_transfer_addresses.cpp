#include "dtx.hpp"
#include "util_vector_of_ints.hpp"
#include "util.hpp"

using namespace std;

bool generate_transfer_addresses(DataTransformations * dtx){
    bool DEBUG = false;

    if (dtx->transformations.size() > 2) throw std::runtime_error("DTX error: your DTX contains more than 2 transformations. First run collapse_transormations(), then generate_transfer_addresses");

    if (DEBUG) tt::log_info(tt::LogDTX, "\n----- Starting to Generate Transfers -----\n");

    TransformationNode * producer_node = dtx->transformations[0];
    TransformationNode * consumer_node = dtx->transformations[1];

    // Loop over all groups
    // TODO: generalize for producer groups > 1
    assert(producer_node->groups.size() == 1);
    int rank = producer_node->groups[0]->shape.size();
    assert(vector_product(producer_node->groups[0]->shape) == producer_node->groups[0]->shape[X(rank)]);
    for (TensorPairGroup * consumer_group : consumer_node->groups) {
        assert(consumer_group->shape.size() == rank);
        // Loop over all TensorPairs
        for (TensorPair * consumer_tp : consumer_group->tensor_pairs) {
            //if (DEBUG) consumer_tp->print_string();

            assert(consumer_tp->src_tensor->str.size() == rank);
            assert(consumer_tp->dst_tensor->str.size() == rank);
            assert(consumer_tp->src_tensor->end.size() == rank);
            assert(consumer_tp->dst_tensor->end.size() == rank);
            assert(vector_product(consumer_group->shape) == consumer_group->shape[X(rank)]);
            Transfer * transfer = new Transfer();
            transfer->src_address = consumer_tp->src_tensor->str[X(rank)] + producer_node->groups[0]->address;
            transfer->dst_address = consumer_tp->dst_tensor->str[X(rank)] + consumer_group->address;
            assert(consumer_tp->src_tensor->volume() == consumer_tp->dst_tensor->volume());
            transfer->size = consumer_tp->src_tensor->volume();
            transfer->pad = consumer_tp->src_group == -1 ? 1 : 0;
            transfer->src_soc_core =  copy_vector_of_ints(producer_node->groups[0]->core);

            consumer_group->transfers.push_back(transfer);

            if (DEBUG) log_info(tt::LogDTX, "{}", transfer->get_string());
        }
    }

    if (DEBUG) dtx->print();

    if (DEBUG) tt::log_info(tt::LogDTX, "\n----- Ending Generate Transfers -----\n");

    return true;
}

bool generate_transfer_addresses_blocked_data(DataTransformations * dtx){
    // used for generating address for data which is stored in 2d blocks like tiled data
    // producer and consumer of dtx transformation should have same block shape

    bool DEBUG = false;

    if (dtx->transformations.size() > 2) throw std::runtime_error("DTX error: your DTX contains more than 2 transformations. First run collapse_transormations(), then generate_transfer_addresses");

    if (DEBUG) tt::log_info(tt::LogDTX, "\n----- Starting to Generate Transfers -----\n");

    TransformationNode * producer_node = dtx->transformations[0];
    TransformationNode * consumer_node = dtx->transformations[1];
    // TODO: generalize for producer groups > 1
    assert(producer_node->groups.size() == 1);
    int rank = producer_node->groups[0]->shape.size();
    assert(rank == 3);
    int block_shape_y = producer_node->groups[0]->shape[1];
    int block_shape_x = producer_node->groups[0]->shape[2];
    // Loop over all groups
    for (TensorPairGroup * consumer_group : consumer_node->groups) {
        assert(consumer_group->shape.size() == rank);
        // consumer and producer should have same block shape
        assert(consumer_group->shape[1] == producer_node->groups[0]->shape[1]);
        assert(consumer_group->shape[2] == producer_node->groups[0]->shape[2]);
        // Loop over all TensorPairs
        for (TensorPair * consumer_tp : consumer_group->tensor_pairs) {
            //if (DEBUG) consumer_tp->print_string();
            assert(consumer_tp->src_tensor->str.size() == rank);
            assert(consumer_tp->dst_tensor->str.size() == rank);
            assert(consumer_tp->src_tensor->end.size() == rank);
            assert(consumer_tp->dst_tensor->end.size() == rank);
            Transfer * transfer = new Transfer();
            assert(consumer_tp->src_tensor->str[1] == 0);
            assert(consumer_tp->src_tensor->str[2] == 0);
            assert(consumer_tp->src_tensor->end[1] == block_shape_y-1);
            assert(consumer_tp->src_tensor->end[2] == block_shape_x-1);
            assert(consumer_tp->dst_tensor->str[1] == 0);
            assert(consumer_tp->dst_tensor->str[2] == 0);
            assert(consumer_tp->dst_tensor->end[1] == block_shape_y-1);
            assert(consumer_tp->dst_tensor->end[2] == block_shape_x-1);
            transfer->src_address = consumer_tp->src_tensor->str[0]*block_shape_y*block_shape_x;
            transfer->dst_address = consumer_tp->dst_tensor->str[0]*block_shape_y*block_shape_x;
            assert(consumer_tp->src_tensor->volume() == consumer_tp->dst_tensor->volume());
            transfer->size = consumer_tp->src_tensor->volume();
            transfer->pad = consumer_tp->src_group == -1 ? 1 : 0;
            assert(consumer_tp->src_tensor->str[1]%32 == 0);
            transfer->src_soc_core =  copy_vector_of_ints(producer_node->groups[0]->core);

            consumer_group->transfers.push_back(transfer);

            if (DEBUG) tt::log_info(tt::LogDTX, "{}",  transfer->get_string());
        }
    }

    if (DEBUG) dtx->print();

    if (DEBUG) tt::log_info(tt::LogDTX, "\n----- Ending Generate Transfers -----");

    return true;
}


bool generate_transfer_addresses_tiled_data(DataTransformations * dtx){
    bool DEBUG = false;

    if (dtx->transformations.size() > 2) throw std::runtime_error("DTX error: your DTX contains more than 2 transformations. First run collapse_transormations(), then generate_transfer_addresses");

    if (DEBUG) tt::log_info(tt::LogDTX, "\n----- Starting to Generate Transfers -----");

    TransformationNode * producer_node = dtx->transformations[0];
    TransformationNode * consumer_node = dtx->transformations[1];

    // Loop over all groups
    for (TensorPairGroup * consumer_group : consumer_node->groups) {
        assert(consumer_group->shape.size() == 2 && consumer_group->shape[0] == 32);
        // Loop over all TensorPairs
        for (TensorPair * consumer_tp : consumer_group->tensor_pairs) {
            //if (DEBUG) consumer_tp->print_string();
            Transfer * transfer = new Transfer();
            assert(consumer_tp->src_tensor->str[0] == 0);
            assert(consumer_tp->src_tensor->str[1]%32 == 0);
            transfer->src_address = (consumer_tp->src_tensor->str[1]/32)*32*32;
            assert(consumer_tp->dst_tensor->str[0] == 0);
            assert(consumer_tp->dst_tensor->str[1]%32 == 0);
            transfer->dst_address = (consumer_tp->dst_tensor->str[1]/32)*32*32;
            assert(consumer_tp->src_tensor->volume() == 32*32);
            assert(consumer_tp->dst_tensor->volume() == consumer_tp->src_tensor->volume());
            transfer->size = consumer_tp->src_tensor->volume();
            transfer->src_soc_core =  copy_vector_of_ints(producer_node->groups[consumer_tp->src_group]->core);

            consumer_group->transfers.push_back(transfer);

            if (DEBUG) tt::log_info(tt::LogDTX, "{}", transfer->get_string());
        }
    }

    if (DEBUG) dtx->print();

    if (DEBUG) tt::log_info(tt::LogDTX, "\n----- Ending Generate Transfers -----");

    return true;
}
