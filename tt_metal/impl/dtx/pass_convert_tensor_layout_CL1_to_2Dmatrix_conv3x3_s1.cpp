#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

/*
vector<vector<int>> dim_order_counting(vector<int> shape, vector<int> dim_order) {
    bool DEBUG = false;
    cout << "\n\nHelper function: dim order counting" << endl;

    int rank = shape.size();
    vector<int> shape_reordered;
    for (int d=0; d<rank; d++) {
        shape_reordered.push_back(shape[dim_order[d]]);
    }
    int count_size = vector_product(shape);
    vector<int> tile_shape = vector_pad_on_left(TILE_SHAPE, rank-2, 1);

    if (DEBUG) {
        cout << "shape           = " << v2s(shape) << endl;
        cout << "dim order       = " << v2s(dim_order) << endl;
        cout << "shape reordered = " << v2s(shape_reordered) << endl;
        cout << "count size      = " << count_size << endl;
        cout << "tile shape      = " << v2s(tile_shape) << endl;
    }

    vector<vector<int>> list_of_counted_dims;
    vector<int> counter = zeros(rank);
    for (int i=0; i<count_size; i++){
        vector<int> counter_reordered;

        for (int d=0; d<rank; d++) {
            counter_reordered.push_back(counter[dim_order[d]]);
        }

        vector<int> str = vector_multiplication(counter_reordered, tile_shape);
        vector<int> end = vector_addition(str, tile_shape);
        end[rank-1]--;
        end[rank-2]--;

        cout << s(3) << "counter = " << v2s(counter) << ", reorderd = " << v2s(counter_reordered) << ";   " << v2s(str) << " => " << v2s(end) << endl;
        list_of_counted_dims.push_back(counter_reordered);

        counter.back()++;
        for (int d=rank-1; d>0; d--) {
            if (counter[d] == shape_reordered[d]) {
                counter[d-1]++;
                counter[d] = 0;
            }
        }
    }
    cout << endl;
    cout << endl;

    return list_of_counted_dims;
}
*/


bool convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1(DataTransformations * dtx) {
    /*
    Notes:
    Should not support groups (I think). This transformation has to be applied before parallelizations.
    What if it's applied to a DRAM swizzled layout - need to think about that.
    */


    bool DEBUG = true;
    if (DEBUG) cout << "\n\nPASS: convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1 - START" << endl;



    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1", producer->groups.size());
    dtx->transformations.push_back(consumer);






    // Setup
    vector<int> shape = producer->groups[0]->shape;
    int rank = shape.size();
    vector<int> channel_shape = vector_pad_on_left({shape[0]}, rank-1, 1);

    int x_size = shape[rank-1];
    int y_size = shape[rank-2];

    if (DEBUG) {
        cout << s(2) << "activation shape = " << v2s(shape) << endl;
        cout << s(2) << "channel shape = " << v2s(channel_shape) << endl;
        cout << s(2) << "x/y_size = " << x_size << ", " << y_size << endl;
    }

    // Kernel Window
    int kernel_size_x = 3;
    int kernel_size_y = 3;
    int activation_size_x = shape[rank-1];
    int activation_size_y = shape[rank-2];

    TensorPairGroup * consumer_group = consumer->groups[0];
    //consumer_group->shape = {, };


    // 2D Matrix destination:
    int matrix2d_x = 0;
    int matrix2d_y = 0;

    // Do the work

    // Sweep over the face of the activation
    for (int y=1; y<y_size-1; y++) {
        for (int x=1; x<x_size-1; x++) {
            if (DEBUG) cout << s(2) << "[y, x] = [" << y << ", " << x << "]" << endl;

            // Sweep over the kernel window size (hardcoded 3x3 right now)
            for (int kernel_y=-1; kernel_y<2; kernel_y++){
                for (int kernel_x=-1; kernel_x<2; kernel_x++){

                    int position_y = y + kernel_y;
                    int position_x = x + kernel_x;

                    if (DEBUG) {
                        if (DEBUG) cout << s(4) << "[pos_y, pos_x] = [" << position_y << ", " << position_x << "]" << endl;
                    }


                    //TensorPair * tp = new TensorPair(new Tensor({str}, {end}),
                    //                                            group_idx,
                    //                                            new Tensor({consumer_str}, {consumer_end}));
                    //consumer_group->tensor_pairs.push_back(tp);

                }
            }


        }
    }


    /*
    for (int group_idx=0; group_idx<producer->groups.size(); group_idx++) {
        cout << "\n\n" << s(2) << "Group = " << group_idx << endl;

        TensorPairGroup * consumer_group = consumer->groups[group_idx];
        TensorPairGroup * producer_group = producer->groups[group_idx];
        inherit_group_attributes_from_producer(producer_group, consumer_group);

        vector<int> shape = producer_group->shape;
        int rank = producer_group->shape.size();
        vector<int> tile_shape = vector_pad_on_left(TILE_SHAPE, rank-2, 1);
        vector<int> shape_tiled = vector_division(shape, tile_shape);
        vector<vector<int>> list_of_counted_dims = dim_order_counting(shape_tiled,   dim_order);
        vector<int> consumer_str = zeros(rank);
        vector<int> consumer_end = vector_addition(consumer_str, tile_shape);
        cout << s(4) << "tile shape      = " << v2s(tile_shape) << endl;

        if (shape.size() != dim_order.size()) throw std::runtime_error("shape and dim_order dont have the same rank!");

        int shape_x = list_of_counted_dims.size() * 32;
        consumer_group->shape = {32, shape_x};

        cout << s(4) << "Tensor Pairs: " << list_of_counted_dims.size() << endl;
        for (int i=0; i< list_of_counted_dims.size(); i++) {

            // Source Tensor: within the ND tensor from producer
            vector<int> str;
            vector<int> end;
            str = vector_multiplication(list_of_counted_dims[i], tile_shape);
            end = vector_addition(str, tile_shape);
            end[rank-1]--;
            end[rank-2]--;

            TensorPair * tp = new TensorPair(new Tensor({str}, {end}),
                                            group_idx,
                                            new Tensor({consumer_str}, {consumer_end}));
            consumer_group->tensor_pairs.push_back(tp);

            cout << s(6) << i << ".  " << tp->get_string() << endl;

            // Prepare for the next itteration
            consumer_str.back() = ((i+1) * 32) - 1;
            consumer_end = vector_addition(consumer_str, tile_shape);
        }
    }
    */


    if (DEBUG) cout << "\n\nPASS: convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1 - END\n\n" << endl;
    return true;
}
