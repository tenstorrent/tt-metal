#include "dtx.hpp"
#include "util_vector_of_ints.hpp"

DataTransformations * reverse_transformations(DataTransformations * fw_dtx) {
    bool DEBUG = true;
    if (DEBUG) fw_dtx->print();
    if (DEBUG) cout << "\n\n ----- Start Reverse Transformations ------" << endl;
    DataTransformations * bck_dtx = new DataTransformations();

    // Step 1: Create a new producer node
    TransformationNode * fw_last_node = fw_dtx->transformations.back();
    TransformationNode * bck_first_node = new TransformationNode("producer", fw_last_node->groups.size());
    bck_dtx->transformations.push_back(bck_first_node);

    for (int g=0; g<bck_first_node->groups.size(); g++){
        vector<int> new_shape(fw_last_node->groups[g]->shape);
        bck_first_node->groups[g]->shape = new_shape;
    }

    // Step 2: For every pair of nodes in the forward dtx, create a new node in the backward dtx
    for (int t = fw_dtx->transformations.size()-1; t>0; t--) {
        cout << "transofmration index: " << t << endl;

        // Setup:  1) identify all the relevant nodes, 2) create new node to be populated
        TransformationNode * fwd_producer = fw_dtx->transformations[t-1];
        TransformationNode * fwd_consumer = fw_dtx->transformations[t];
        TransformationNode * bck_producer = bck_dtx->transformations.back();
        string new_opcode = fwd_consumer->opcode;
        new_opcode.append("_reversed");
        TransformationNode * bck_consumer = new TransformationNode(new_opcode, fwd_producer->groups.size());
        bck_dtx->transformations.push_back(bck_consumer);

        // Assign shapes to groups in bck_consumer
        for (int fwd_producer_group_idx=0; fwd_producer_group_idx<fwd_producer->groups.size(); fwd_producer_group_idx++) {
            vector<int> new_shape(fwd_producer->groups[fwd_producer_group_idx]->shape);
            bck_consumer->groups[fwd_producer_group_idx]->shape = copy_vector_of_ints(fwd_producer->groups[fwd_producer_group_idx]->shape);
        }
        
        // Reverse all the TensorPairs from fwd_consumer and put them into bck_consumer
        int g=0;
        for (TensorPairGroup * fwd_consumer_group : fwd_consumer->groups) {
            for (TensorPair * fwd_consumer_tp : fwd_consumer_group->tensor_pairs) {
                
                Tensor * bck_src_tensor = new Tensor(copy_vector_of_ints(fwd_consumer_tp->dst_tensor->str), copy_vector_of_ints(fwd_consumer_tp->dst_tensor->end));
                Tensor * bck_dst_tensor = new Tensor(copy_vector_of_ints(fwd_consumer_tp->src_tensor->str), fwd_consumer_tp->src_tensor->end);
                int bck_src_group = g;
                
                TensorPair * bck_tp = new TensorPair(bck_src_tensor, bck_src_group, bck_dst_tensor);
                bck_consumer->groups[fwd_consumer_tp->src_group]->tensor_pairs.push_back(bck_tp);
            }
            g++;
        }    
    }

    if (DEBUG) cout << " ----- End Reverse Transformations ------\n\n" << endl;
    if (DEBUG) bck_dtx->print();
    return bck_dtx;
}
