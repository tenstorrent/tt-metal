#include "dtx.hpp"
#include "util_vector_of_ints.hpp"
#include "util.hpp"

bool generate_transfer_addresses(DataTransformations * dtx){
    bool DEBUG = false;

    if (dtx->transformations.size() > 2) throw std::runtime_error("DTX error: your DTX contains more than 2 transformations. First run collapse_transormations(), then generate_transfer_addresses");

    if (DEBUG) cout << "\n----- Starting to Generate Transfers -----\n" << endl;

    TransformationNode * producer_node = dtx->transformations[0];
    TransformationNode * consumer_node = dtx->transformations[1];

    // Loop over all groups
    // TODO: generalize for groups > 1
    assert(consumer_node->groups.size() == 1);
    for (TensorPairGroup * consumer_group : consumer_node->groups) {

        // Loop over all TensorPairs
        for (TensorPair * consumer_tp : consumer_group->tensor_pairs) {
            //if (DEBUG) consumer_tp->print_string();

            int rank = consumer_tp->src_tensor->str.size();

            Transfer * transfer = new Transfer();
            transfer->src_address = consumer_tp->src_tensor->str[X(rank)] + producer_node->groups[0]->address;
            transfer->dst_address = consumer_tp->dst_tensor->str[X(rank)] + consumer_group->address;
            transfer->size = consumer_tp->src_tensor->volume();
            transfer->pad = consumer_tp->src_group == -1 ? 1 : 0;
            transfer->src_soc_core =  copy_vector_of_ints(producer_node->groups[0]->core);

            consumer_group->transfers.push_back(transfer);

            if (DEBUG) cout << transfer->get_string() << endl;
        }
    }

    if (DEBUG) dtx->print();

    if (DEBUG) cout << "\n----- Ending Generate Transfers -----\n" << endl;

    return true;
}

bool generate_transfer_addresses_tiled_data(DataTransformations * dtx){
    bool DEBUG = false;

    if (dtx->transformations.size() > 2) throw std::runtime_error("DTX error: your DTX contains more than 2 transformations. First run collapse_transormations(), then generate_transfer_addresses");

    if (DEBUG) cout << "\n----- Starting to Generate Transfers -----\n" << endl;

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

            if (DEBUG) cout << transfer->get_string() << endl;
        }
    }

    if (DEBUG) dtx->print();

    if (DEBUG) cout << "\n----- Ending Generate Transfers -----\n" << endl;

    return true;
}
