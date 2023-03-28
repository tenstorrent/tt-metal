#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"
#include "dtx_passes.hpp"


bool convert_tensor_layout_CL1_to_2Dmatrix_conv1x1_s1(DataTransformations * dtx) {
    /*
    Notes:
    Should not support groups (I think). This transformation has to be applied before parallelizations.
    What if it's applied to a DRAM swizzled layout - need to think about that.

    Note: Input activatins need to already be padded. Padding not supported yet.
    // int pad = 1; // all around
    */

    bool DEBUG = true;
    bool pass = true;
    if (DEBUG) cout << "\n\nPASS: convert_tensor_layout_CL1_to_2Dmatrix_conv1x1_s1 - START" << endl;

    // First add the 2 required transposes
    pass &= transpose_yz(dtx);
    pass &= transpose_xy(dtx);

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("convert_tensor_layout_CL1_to_2Dmatrix_conv1x1_s1", producer->groups.size());
    dtx->transformations.push_back(consumer);

    // Calculate producer shape
    vector<int> producer_shape = producer->groups[0]->shape;
    int rank = producer_shape.size();

    // Calculate the consumer shape
    int kernel_x = 1;
    int kernel_y = 1;
    int consumer_shape_x = kernel_x * kernel_y * producer_shape[X(rank)];
    int consumer_shape_y =  producer_shape[Y(rank)]  *  producer_shape[Z(rank)];
    //vector<int> consumer_shape = {consumer_shape_y, consumer_shape_x};
    consumer->groups[0]->shape = {1, consumer_shape_y, consumer_shape_x};

    int consumer_y = 0;
    int consumer_x = 0;
    for (int producer_y=0; producer_y<producer_shape[Y(rank)]; producer_y++) {
        for (int producer_z=0; producer_z<producer_shape[Z(rank)]; producer_z++) {

            if (DEBUG) cout << endl;
            if (DEBUG) cout << s(2) << "producer_y/z = " << producer_z << "," << producer_y << endl;


            // Producer str/end
            vector<int> producer_str = { producer_z, producer_y, 0};
            vector<int> producer_end = { producer_z, producer_y, producer_shape[X(rank)]-1};

            vector<int> consumer_str = {0, consumer_y, consumer_x};
            vector<int> consumer_end = {0, consumer_y, consumer_x + producer_shape[X(rank)]-1};

            TensorPair * tp = new TensorPair(new Tensor({producer_str}, {producer_end}),
                                            0,
                                            new Tensor({consumer_str}, {consumer_end}));
            consumer->groups[0]->tensor_pairs.push_back(tp);

            if (DEBUG) cout << s(6) << "src = " << v2s(producer_str) << "-" << v2s(producer_end) << " ==> " << v2s(consumer_str) << "-" << v2s(consumer_end) << endl;

            consumer_y++;
        }
    }




    if (DEBUG) cout << "\n\nPASS: convert_tensor_layout_CL1_to_2Dmatrix_conv1x1_s1 - END\n\n" << endl;
    return true;
}
