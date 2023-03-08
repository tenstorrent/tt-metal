#include "tt_metal/host_api.hpp"
#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "constants.hpp"


#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

using namespace std;
using namespace tt;

using namespace constants;


int covert_shape_index_to_rowmajor_index(vector<int> shape_index, vector<int> shape){
    bool DEBUG = false;

    int rank = shape.size();
    int rm_index = 0;

    if (DEBUG) {
        cout << endl;
        cout << s(8) << "Index conversion:" << endl;
        cout << s(8) << "shape = " << v2s(shape) << ", index = " << v2s(shape_index) << endl;
        cout << s(8) << "rank = " << rank << endl;
    }

    // Rank is 1 (nothing to do)
    if (rank == 1) {
        return shape_index[0];
    }

    // Rank > 1
    int dim_volume = 1;
    for (int d=rank-1; d>-1; d--) {
        if (DEBUG) cout << endl;
        if (DEBUG) cout << s(8) << "d = " << d << ", shape dim = " << shape[d] << endl;
        if (DEBUG) cout << s(10) << "dim_volume = " << dim_volume << endl;
        rm_index += shape_index[d] * dim_volume;
        if (DEBUG) cout << s(10) << "current rm index = " << rm_index << endl;
        dim_volume = dim_volume * shape[d];
    }
    cout << s(8) << "final index = " << rm_index << endl;
    return rm_index;
}



void evaluate(vector<uint32_t> src_data, DataTransformations * dtx) {
    cout <<"\nEvaluate demo\n" << endl;


    //cout << data[3] << endl;
    //cout << data.size() << endl;


    /*
    cout << "\n\nconvert: " << covert_shape_index_to_rowmajor_index({2, 5}, {10, 10}) << endl;
    cout << "\n\nconvert: " << covert_shape_index_to_rowmajor_index({3, 4, 5}, {12,11,10}) << endl;
    cout << "\n\nconvert: " << covert_shape_index_to_rowmajor_index({3, 4, 5}, {100,10,10}) << endl;
    cout << "\n\nconvert: " << covert_shape_index_to_rowmajor_index({3, 4, 5}, {10,100,10}) << endl;
    */

    // Tensor(std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, Layout layout)

    //tt_metal::Tensor out = tt_metal::Tensor({1,1,1,6}, tt_metal::Initialize::INCREMENT, tt_metal::Layout::ROW_MAJOR);

    vector<uint32_t> dst_data;
    for (int i=0; i<src_data.size(); i++) {
        dst_data.push_back(99);
    }

    for (auto tp : dtx->transformations[1]->groups[0]->tensor_pairs) {
        Tensor * src_tensor = tp->src_tensor;
        Tensor * dst_tensor = tp->dst_tensor;
        vector<int> src_shape = dtx->transformations[0]->groups[0]->shape;
        vector<int> dst_shape = dtx->transformations[1]->groups[0]->shape;

        cout << "src: " << v2s(src_tensor->str) << " --> " << v2s(src_tensor->end) << endl;
        cout << "dst: " << v2s(dst_tensor->str) << " --> " << v2s(dst_tensor->end) << endl;

        vector<int> src_index = src_tensor->str;
        vector<int> dst_index = dst_tensor->str;

        dst_data[covert_shape_index_to_rowmajor_index(dst_index, dst_shape )] = src_data[covert_shape_index_to_rowmajor_index(src_index, src_shape)];
        while (!compare_two_vectors_of_ints(src_index, src_tensor->end)) {
            cout << s(2) << v2s(src_index) << " --> " << v2s(dst_index) << endl;
            src_index = increment(src_index, src_tensor->str, src_tensor->end);
            dst_index = increment(dst_index, dst_tensor->str, dst_tensor->end);
            dst_data[covert_shape_index_to_rowmajor_index(dst_index, dst_shape )] = src_data[covert_shape_index_to_rowmajor_index(src_index, src_shape)];
        }

    }

    cout << "\nsrc_data = ";
    for (int i=0; i<src_data.size(); i++) {
        cout << src_data[i] << ", ";
    }
    cout << endl;

    cout << "\ndst_data = ";
    for (int i=0; i<dst_data.size(); i++) {
        cout << dst_data[i] << ", ";
    }
    cout << endl;



}
