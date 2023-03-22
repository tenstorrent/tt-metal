#include "tt_metal/host_api.hpp"
#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "constants.hpp"

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/impl/dtx/dtx.hpp"
#include "tt_metal/impl/dtx/dtx_passes.hpp"

using namespace tt;

using namespace constants;


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;


    std::array<uint32_t, 4> shape = {1, 1, 1, 6};
    tt_metal::Tensor a = tt_metal::Tensor(shape, tt_metal::Initialize::INCREMENT, tt_metal::Layout::ROW_MAJOR);
    a.print();
    a.test();


    DataTransformations * dtx = new DataTransformations();

    // Base Node
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {1,1,1,6};
    dtx->transformations.push_back(node0);

    // 1 transformation
    TransformationNode * node1 = new TransformationNode("transformation", 1);
    node1->groups[0]->shape = {1,1,1,6};
    dtx->transformations.push_back(node1);
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0,0,0,0},  {0,0,0,2}), 0,  new Tensor({0,0,0,3},  {0,0,0,5}))   );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0,0,0,3},  {0,0,0,5}), 0,  new Tensor({0,0,0,0},  {0,0,0,2}))   );

    dtx->print();
    evaluate(a.to_vec(), dtx);

    return 1;
}
