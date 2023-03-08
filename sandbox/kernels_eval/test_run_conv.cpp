
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <array>
#include <cassert>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <chrono>
#include <cmath>
#include "tensor/tensor.hpp"
#include "conv_pattern.hpp"
using namespace std;

enum INITIALIZE
{
    Zeros = 0,
    Ones = 1,
    Increment = 2,
    Random = 3,
    Value = 4
};

/* class ConvParameters {
    public:
        uint32_t stride_x = 1;
        uint32_t stride_y = 1;
        std::array<uint32_t, 2> padding_x = {0, 0};
        std::array<uint32_t, 2> padding_y = {0, 0};
        ConvParameters() {}
        ConvParameters(uint32_t stride_x_, uint32_t stride_y_, std::array<uint32_t, 2> padding_x_, std::array<uint32_t, 2> padding_y_) :
        stride_x(stride_x_), stride_y(stride_y_), padding_x(padding_x_), padding_y(padding_y_) {}
        void print() {
            std::cout << "stride x = " << stride_x << " , stride y = " << stride_y << std::endl;
            std::cout << "left padding = " << padding_x[0] << " , right padding = " << padding_x[1] << std::endl;
            std::cout << "top padding = " << padding_y[0] << " , bottom padding = " << padding_y[1] << std::endl;
        }
};

class Tensor {
    public:
        Tensor(std::vector<double> &values, std::array<uint32_t, 4> &shape) {
            this->shape = shape;
            this->values = values;
            this->strides = {shape[1]*shape[2]*shape[3], shape[2]*shape[3], shape[3], 1};
        }
        Tensor(std::array<uint32_t, 4> &shape) {
            this->shape = shape;
            auto volume = shape[0] * shape[1] * shape[2] * shape[3];
            this->values.resize(volume, 0);
            this->strides = {shape[1]*shape[2]*shape[3], shape[2]*shape[3], shape[3], 1};
        }
        vector<double> get_values() {
            return this->values;
        }
        void print() {
            std::cout<<"Shape = ["<<shape[0]<<","<<shape[1]<<","<<shape[2]<<","<<shape[3]<<"]"<<std::endl;
            std::cout<<"Strides = ["<<strides[0]<<","<<strides[1]<<","<<strides[2]<<","<<strides[3]<<"]"<<std::endl;
            std::cout<<"Values = [";
            for(auto w = 0; w < shape[0]; w++) {
                std::cout<<"[";
                for(auto z = 0; z < shape[1]; z++) {
                    std::cout<<"[";
                    for(auto y = 0; y < shape[2]; y++) {
                        std::cout<<"[";
                        for(auto x = 0; x < shape[3]; x++) {
                            auto idx = x + shape[3] * y + shape[3] * shape[2] * z + shape[3] * shape[2] * shape[1] * w;
                            std::cout<<values[idx]<<",";
                        }
                        std::cout<<"]"<<std::endl;
                    }
                    std::cout<<"]"<<std::endl;
                }
                std::cout<<"]"<<std::endl;
            }
            std::cout<<"]"<<std::endl;
        }
    // private:
        vector<double> values;
        std::array<uint32_t, 4> shape; // outer-most dimension first
        std::array<uint32_t, 4> strides; // outer-most dimension first
};
 */

/* Tensor get_tensor(std::array<uint32_t, 4> &shape, INITIALIZE init_type) {
    srand (0);
    std::vector<double> values;
    for(auto w = 0; w < shape[0]; w++) {
        for(auto z = 0; z < shape[1]; z++) {
            for(auto y = 0; y < shape[2]; y++) {
                for(auto x = 0; x < shape[3]; x++) {
                    double val;
                    switch (init_type)
                    {
                        case INITIALIZE::Zeros:
                            val = 0;
                            break;
                        case INITIALIZE::Ones:
                            val = 1;
                            break;
                        case INITIALIZE::Value:
                            val = 0.5;
                            break;
                        case INITIALIZE::Increment:
                            val = x + shape[3] * y + shape[3] * shape[2] * z + shape[3] * shape[2] * shape[1] * w;
                            break;
                        case INITIALIZE::Random:
                            val = ((double) rand() / (RAND_MAX));
                            break;
                        default:
                            break;
                    }
                    values.push_back(val);
                }
            }
        }
    }

    return Tensor(values, shape);
}
 */
template <typename T>
tt::Tensor<T> conv(tt::Tensor<T> &activation, tt::Tensor<T> &weights, ConvParameters& conv_params) {
    std::array<uint32_t, 4> act_shape = activation.get_shape();
    std::array<uint32_t, 4> weights_shape = weights.get_shape();
    auto activation_values = activation.get_values();
    auto weights_values = weights.get_values();
    std::array<uint32_t, 4> out_shape = {
                                        act_shape[0],
                                        weights_shape[0],
                                        ( (act_shape[2] + 2 * conv_params.PadH - weights_shape[2]) / conv_params.U) + 1,
                                        ( (act_shape[3] + 2 * conv_params.PadW - weights_shape[3]) / conv_params.V) + 1
                                        };
    auto output_volume = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    std::vector<T> out = std::vector<T>(output_volume, 0);
    // int padding = weights_shape[2] / 2;
    int filter_center_x = weights_shape[3] / 2;
    int filter_center_y = weights_shape[2] / 2;
    for(auto w = 0; w < out_shape[0]; w++) { // N
        for(auto z = 0; z < out_shape[1]; z++) { // Cout
            for(auto y = 0; y < out_shape[2]; y++) { // Y
                for(auto x = 0; x < out_shape[3]; x++) { // X
                    // output flat idx
                    auto idx = x + out_shape[3] * y + out_shape[3] * out_shape[2] * z + out_shape[3] * out_shape[2] * out_shape[1] * w;
                    T result = 0;
                    // std::cout<<"Y="<<y<<", X="<<x<<std::endl;
                    for(auto fx = 0; fx < weights_shape[3]; fx++) { // filter x
                        for(auto fy = 0; fy < weights_shape[2]; fy++) { // filter y
                            for(auto cin = 0; cin < act_shape[1]; cin++) { // filter cin
                                // auto act_x = x-(filter_center_x - fx);
                                // auto act_y = y-(filter_center_y - fy);
                                auto act_x = x * conv_params.V - conv_params.PadW + fx;
                                auto act_y = y * conv_params.U - conv_params.PadH + fy;
                                // std::cout<<"act_y = "<<act_y<<", act_x = "<<act_x<<std::endl;
                                // std::cout<<"fy = "<<fy<<", fx = "<<fx<<std::endl;
                                if(act_x>=0 and act_x < act_shape[3] and act_y >=0 and act_y < act_shape[2]){
                                    auto act_idx = act_x + act_y * act_shape[3] + cin * act_shape[2] * act_shape[3] + w * act_shape[1]*act_shape[2]*act_shape[3];
                                    auto weights_idx = fx + fy * weights_shape[3] + cin * weights_shape[2] * weights_shape[3] + z * weights_shape[1]*weights_shape[2]*weights_shape[3];
                                    // std::cout<<activation.values[act_idx]<<", "<<weights.values[weights_idx]<<std::endl;
                                    result += (activation_values[act_idx] * weights_values[weights_idx]);
                                }
                            }
                        }
                    }
                    out[idx] = result;
                    // std::cout<<std::endl<<std::endl;
                }
            }
        }
    }
    return tt::Tensor<T>(out, out_shape);
}

/* Tensor convert_XYZW_to_ZXYW(Tensor &input) {
    std::array<uint32_t, 4> in_shape = input.shape;
    std::array<uint32_t, 4> out_shape = {in_shape[0], in_shape[2], in_shape[3], in_shape[1]};
    auto output_volume = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    std::vector<double> out = std::vector<double>(output_volume, 0);

    for(auto w = 0; w < in_shape[0]; w++) { // N
        for(auto z = 0; z < in_shape[1]; z++) { // Z
            for(auto y = 0; y < in_shape[2]; y++) { // Y
                for(auto x = 0; x < in_shape[3]; x++) { // X
                    auto in_idx = x + y * in_shape[3] + z * in_shape[3] * in_shape[2] + w * in_shape[3] * in_shape[2] * in_shape[1];
                    auto out_idx = z + x * out_shape[3] + y * out_shape[3] * out_shape[2] + w * out_shape[3] * out_shape[2] * out_shape[1];
                    out[out_idx] = input.values[in_idx];
                }
            }
        }
    }

    return Tensor(out, out_shape);
}
 */

/* Tensor convert_ZXYW_to_XYZW(Tensor &input) {
    std::array<uint32_t, 4> in_shape = input.shape;
    std::array<uint32_t, 4> out_shape = {in_shape[0], in_shape[3], in_shape[1], in_shape[2]};
    auto output_volume = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    std::vector<double> out = std::vector<double>(output_volume, 0);

    for(auto w = 0; w < out_shape[0]; w++) { // N
        for(auto z = 0; z < out_shape[1]; z++) { // Z
            for(auto y = 0; y < out_shape[2]; y++) { // Y
                for(auto x = 0; x < out_shape[3]; x++) { // X
                    auto out_idx = x + y * out_shape[3] + z * out_shape[3] * out_shape[2] + w * out_shape[3] * out_shape[2] * out_shape[1];
                    auto in_idx = z + x * in_shape[3] + y * in_shape[3] * in_shape[2] + w * in_shape[3] * in_shape[2] * in_shape[1];
                    out[out_idx] = input.values[in_idx];
                }
            }
        }
    }

    return Tensor(out, out_shape);
}

 */

/* vector<vector<double>> move_act_dram_to_l1(Tensor &input_zxyw, std::array<uint32_t, 4> weights_shape, ConvParameters& conv_params) {
    vector<vector<double>> output;
    std::array<uint32_t, 4> in_shape = input_zxyw.shape;
    std::array<uint32_t, 2> out_shape_face_xy = {
                                        ( (in_shape[1] + conv_params.padding_y[0] + conv_params.padding_y[1] - weights_shape[2]) / conv_params.stride_y) + 1,
                                        ( (in_shape[2] + conv_params.padding_x[0] + conv_params.padding_x[1] - weights_shape[3]) / conv_params.stride_x) + 1
                                        };

    std::vector<std::pair<int, int>> increments;
    for (auto j=0; j<weights_shape[2]; j++) {
        for(auto i =0; i<weights_shape[3]; i++) {
            increments.push_back(std::make_pair(i, j));
        }
    }

    for(auto w = 0; w < in_shape[0]; w++) {
        for(auto y = 0; y < out_shape_face_xy[0]; y++) {
            for(auto x = 0; x < out_shape_face_xy[1]; x++) {
                std::vector<double> row;
                // cout<<"X = "<<x<<" Y = "<<y<<endl;
                for(auto increment: increments) {
                    auto x_new = x * conv_params.stride_x - conv_params.padding_x[0] + increment.first;
                    auto y_new = y * conv_params.stride_y - conv_params.padding_y[0] + increment.second;
                    // cout<<"\tXnew = "<<x_new<<" Ynew = "<<y_new<<endl;
                    for(auto z = 0; z < in_shape[3]; z++) {
                        if(x_new < 0 or y_new < 0 or x_new >= in_shape[2] or y_new >= in_shape[1]) {
                            row.push_back(0);
                        } else {
                            auto idx = z + x_new * in_shape[3] + y_new * in_shape[3] * in_shape[2] + w * in_shape[3] * in_shape[2] * in_shape[1];
                            assert(idx >= 0 and idx < input_zxyw.values.size());
                            row.push_back(input_zxyw.values[idx]);
                        }
                    }
                }
                output.push_back(row);
            }
        }

    }

    return output;
}
 */
/* vector<vector<double>> move_weights_dram_to_l1(Tensor &input_zxyw) {
    vector<vector<double>> output;
    std::array<uint32_t, 4> in_shape = input_zxyw.shape;


    for(auto w = 0; w < in_shape[0]; w++) {
        std::vector<double> row;
        for(auto y = 0; y < in_shape[1]; y++) {
            for(auto x = 0; x < in_shape[2]; x++) {
                for(auto z = 0; z < in_shape[3]; z++) {
                    auto idx = z + x * in_shape[3] + y * in_shape[3] * in_shape[2] + w * in_shape[3] * in_shape[2] * in_shape[1];
                    row.push_back(input_zxyw.values[idx]);
                }
            }
        }
        output.push_back(row);
    }
    return output;
} */

template <typename T>
tt::Tensor<T> move_output_from_l1_to_dram(vector<vector<T>> &output, std::array<uint32_t, 4> shape) {
    vector<T> values;
    for(auto r = 0; r < output.size(); r++) {
        for(auto c = 0; c < output[0].size(); c++){
            values.push_back(output[r][c]);
        }
    }
    return tt::Tensor<T>(values, shape);
}

template <typename T>
vector<vector<T>> matmult(vector<vector<T>> &act, vector<vector<T>> &weights) {
    vector<vector<T>> output;
    auto num_filters = weights.size();
    auto cols = weights[0].size();
    for(auto row_idx = 0; row_idx < act.size(); row_idx++) {
        std::vector<T> output_row;
        for(auto f = 0; f < num_filters; f++) {
            T res = 0;
            for(auto c = 0; c < cols; c++) {
                // cout<<act[row_idx][c]<<", "<<weights[f][c]<<endl;
                res += act[row_idx][c] * weights[f][c];
            }
            // cout<<"Res = "<<res<<endl;
            output_row.push_back(res);
        }
        output.push_back(output_row);
    }

    return output;
}


void run_conv(std::array<uint32_t, 4> act_shape, std::array<uint32_t, 4> weights_shape, ConvParameters& conv_params) {

    /********************************************************************* GOLDEN REFERENCE *******************************************************************/
    //Tensor activation = get_tensor(act_shape, INITIALIZE::Random);
    tt::Tensor<std::uint32_t> activation = tt::initialize_tensor<std::uint32_t>(act_shape, tt::Initialize::RANDOM, std::chrono::system_clock::now().time_since_epoch().count());
    //Tensor weights = get_tensor(weights_shape, INITIALIZE::Random);
    tt::Tensor<std::uint32_t> weights = tt::initialize_tensor<std::uint32_t>(weights_shape, tt::Initialize::RANDOM, std::chrono::system_clock::now().time_since_epoch().count());
    // activation.print();
    // weights.print();
    tt::Tensor<std::uint32_t> golden = conv(activation, weights, conv_params);
    // golden.print();

    /*************************************************************************** HOST PREP *******************************************************************/
    std::array<std::array<uint32_t, 2>, 4> pad_size = {{{0, 0}, {0, 0}, {conv_params.PadH, conv_params.PadH}, {conv_params.PadW, conv_params.PadW}}};
    tt::Tensor<std::uint32_t> activation_tensor_padded = tt::pad(activation, pad_size);
    auto activation_nhwc = tt::permute(activation_tensor_padded, {0, 2, 3, 1}); // NHWC
    auto weights_nhwc = tt::permute(weights, {0, 2, 3, 1}); // NHWC
    auto golden_nhwc = tt::permute(golden, {0, 2, 3, 1}); // NHWC
    //Tensor activation_zxyw = convert_XYZW_to_ZXYW(activation);
    //Tensor weights_zxyw = convert_XYZW_to_ZXYW(weights);
    //Tensor golden_zxyw = convert_XYZW_to_ZXYW(golden);
    /*************************************************************************** Evaluation *******************************************************************/
    // Step 1: move A/W from DRAM to L1
    // This will create the 2D matrix by modeling what dram to l1 read patterns are
    auto activation_matrix = move_act_dram_to_l1(activation_nhwc, conv_params);
    //vector<vector<double>> activation_matrix = move_act_dram_to_l1(activation_zxyw, weights_shape, conv_params);

    std::tuple<uint32_t, uint32_t, std::vector<uint32_t>> addr_map = gen_address_map_conv_act_for_mm(activation_nhwc.get_shape(), conv_params);

    auto dram_to_l1_tilized_address_map = std::get<2>(addr_map);
    auto activation_matrix_tilized = move_act_dram_to_l1_tilized(activation_nhwc, dram_to_l1_tilized_address_map);
    //vector<vector<uint32_t>> untilized_act_matrix = untilize_act(activation_matrix_tilized, activation_matrix.size(), activation_matrix.at(0).size());
    //assert(untilized_act_matrix == activation_matrix);
    vector<vector<uint32_t>> weights_matrix = move_weights_dram_to_l1(weights_nhwc);
    // STep 2: execute matmult
    vector<vector<uint32_t>> output_matrix = matmult(activation_matrix, weights_matrix);
    // Step 3: move output from L1 to DRAM (ZXYW layout)
    tt::Tensor<std::uint32_t> output_nhwc = move_output_from_l1_to_dram(output_matrix, golden_nhwc.get_shape());
    std::vector<std::uint32_t> golden_nhwc_values = golden_nhwc.get_values();
    std::vector<std::uint32_t> output_nhwc_values = output_nhwc.get_values();
    if (golden_nhwc_values.size() != golden_nhwc_values.size()) {
        throw std::runtime_error(string("Size of output in nhwc layout does not match the size of golden output in nhwc layout"));
    }
    if (golden_nhwc_values != output_nhwc_values) {
        throw std::runtime_error(string("Output in nhwc layout does not match golden output in nhwc layout"));
    }
    /*************************************************************************** Back to HOST *******************************************************************/
    auto output = tt::permute_nhwc_to_nchw(output_nhwc); // NHWC
    //Tensor output = convert_ZXYW_to_XYZW(output_zxyw);
    std::vector<std::uint32_t> golden_values = golden.get_values();
    std::vector<std::uint32_t> output_values = output.get_values();
    if(output_values.size() != golden_values.size()) {
        throw std::runtime_error(string("Size of output in nchw layout does not match size of golden output in nchw layout"));
    }
    if (golden_values != output_values) {
        for(int i = 0; i < golden_values.size(); i++) {
            if (golden_values[i] != output_values[i] && i < 15) {
                std::cout << "Error at i - " << i << " Golden - " << golden_values[i] << " Output - " << output_values[i] << std::endl;
            }
        }
        throw std::runtime_error(string("Output in nchw layout does not match golden output in nchw layout"));
    }

    //cout<<"Test PASSED!"<<endl;
}

class ConvTestParameters {
    public:
        std::array<uint32_t, 4> act_shape;
        std::array<uint32_t, 4> weights_shape;
        ConvParameters conv_params;
        ConvTestParameters(std::array<uint32_t, 4> act_shape_, std::array<uint32_t, 4> weights_shape_, ConvParameters conv_params_) :
            act_shape(act_shape_), weights_shape(weights_shape_), conv_params(conv_params_) {}
        void print() {
            std::cout<<"Activation Shape = ["<<act_shape[0]<<","<<act_shape[1]<<","<<act_shape[2]<<","<<act_shape[3]<<"]"<<std::endl;
            std::cout<<"Weights Shape = ["<<weights_shape[0]<<","<<weights_shape[1]<<","<<weights_shape[2]<<","<<weights_shape[3]<<"]"<<std::endl;
            conv_params.print();
        }
};

int main(int argc, char** argv)
{

    vector<ConvTestParameters> sweep_conv_test_params;
    sweep_conv_test_params.push_back(ConvTestParameters({1, 32, 10, 10}, {16, 32, 3, 3}, ConvParameters(3, 3, 1, 1, 0, 0)));
    //vector<ConvParameters> sweep_conv_params;
    //sweep_conv_params.push_back(ConvParameters(1, 1, ));
    // Sweep conv parameters -
    // stride x and y - 1 to 3
    // uneven stride x and y - 1 to 3
    // padding left and right 0 to 2
    // padding top and bottom 0 to 2
    // uneven padding left/right and top/bottom 0 to 2
/*     for (uint32_t stride_x = 1; stride_x < 4; stride_x++) {
        for (uint32_t stride_y = 1; stride_y < 4; stride_y++) {
            for (uint32_t padding_left = 0; padding_left < 3; padding_left++) {
                for (uint32_t padding_right = 0; padding_right < 3; padding_right++) {
                    for (uint32_t padding_top = 0; padding_top < 3; padding_top++) {
                        for (uint32_t padding_bottom = 0; padding_bottom < 3; padding_bottom++) {
                            sweep_conv_params.push_back(
                                ConvParameters(stride_x, stride_y, {padding_left, padding_right}, {padding_top, padding_bottom})
                                );
                        }
                    }
                }
            }
        }
    } */
    // Sweep different activation and kernel sizes for each conv parameter
    // vector<uint32_t> sweep_w = {1};//, 16};
    // //vector<std::array<uint32_t, 3>> sweep_xyz = { {32, 32, 64}, {256, 256, 16}, {32, 55, 16}, {55, 32, 16}};
    // //vector<pair<uint32_t, uint32_t>> sweep_kernel_xy = { {1,1}, {3,3}, {1,3}, {3,1}, {5,5}};
    // vector<std::array<uint32_t, 3>> sweep_xyz = { {4, 8, 32} };//, {256, 256, 16}, {32, 55, 16}, {55, 32, 16}};
    // vector<pair<uint32_t, uint32_t>> sweep_kernel_xy = {  {3,3} };//, {1,1}, {1,3}, {3,1}, {5,5}};
    // vector<uint32_t> sweep_num_filters = {16};//, 32};
    // for (auto w : sweep_w) {
    //     for (auto o : sweep_num_filters) {
    //         for (auto [x, y, z] : sweep_xyz) {
    //             for (auto [kernel_x, kernel_y] : sweep_kernel_xy) {
    //                 for (auto conv_params : sweep_conv_params) {
    //                 sweep_conv_test_params.push_back(ConvTestParameters({w, z, y, x}, {o, z, kernel_y, kernel_x}, conv_params));
    //                 }
    //             }
    //         }
    //     }
    // }
/*     std::cout << "Starting conv tests. Going to sweep " << sweep_conv_params.size() << " combinations of conv parameters with - " << std::endl;
    std::cout << "     stride (x/y - 1 to 3) and padding (left/right/top/bottom - 0 to 2)" << std::endl;
    std::cout << " for different activation and kernel sizes. Number of tests in total = " << sweep_conv_test_params.size() << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------------" << std::endl; */
    std::cout << "Total number of tests in total = " << sweep_conv_test_params.size() << std::endl;
    int failing_tests = 0;
    for(auto test_params : sweep_conv_test_params) {
        try{
            run_conv(test_params.act_shape, test_params.weights_shape, test_params.conv_params);
        }
        catch (const runtime_error& error) {
            failing_tests++;
            std::cout << "Test failed" << std::endl;
            std::cout << error.what() << std::endl;
            std::cout << "Conv Test Parameters - " << std::endl;
            test_params.print();
            std::cout << "-------------------------------------------------" << std::endl;
        }
    }
    if (failing_tests == 0) {
        cout << "All tests passed!" << endl;
    }
    else {
        cout<< failing_tests << " out of " << sweep_conv_test_params.size() << " tests failed." <<endl;
        assert(false);
    }
    return 0;
}
