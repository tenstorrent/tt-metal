#include "dtx.hpp"
#include "util_vector_of_ints.hpp"

bool generate_transfer_addresses(DataTransformations * dtx){
    bool DEBUG = true;

    if (dtx->transformations.size() > 2) throw std::runtime_error("DTX error: your DTX contains more than 2 transformations. First run collapse_transormations(), then generate_transfer_addresses");

    if (DEBUG) cout << "\n----- Starting to Generate Transfers -----\n" << endl;

    TransformationNode * producer_node = dtx->transformations[0];
    TransformationNode * consumer_node = dtx->transformations[1];

    // Loop over all groups
    for (TensorPairGroup * consumer_group : consumer_node->groups) {

        // Loop over all TensorPairs
        for (TensorPair * consumer_tp : consumer_group->tensor_pairs) {
            //if (DEBUG) consumer_tp->print_string();

            Transfer * transfer = new Transfer();
            transfer->src_address = consumer_tp->src_tensor->str[0] + producer_node->groups[consumer_tp->src_group]->address;
            transfer->dst_address = consumer_tp->dst_tensor->str[0] + consumer_group->address;
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
