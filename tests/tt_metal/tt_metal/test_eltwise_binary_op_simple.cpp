#include <algorithm>
#include <functional>
#include <random>
#include <iostream>

#include "tt_metal/host_api.hpp"
#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/op_library/eltwise_binary/add.hpp"

using namespace tt;
using namespace std;
using namespace tt::tt_metal;

int main(int argc, char **argv) {

    cout << "\nRunning Test\n" << endl;

    Device *device = CreateDevice(tt::ARCH::GRAYSKULL, 0);
    bool pass = InitializeDevice(device);

    Tensor a = Tensor({1, 1, 32, 32}, Initialize::RANDOM, Layout::TILES, device);
    Tensor b = Tensor({1, 1, 32, 32}, Initialize::ZEROS,  Layout::TILES, device);
    Tensor c = add(a, b);
    Tensor d = c.to(Location::HOST);

    pass &= CloseDevice(device);
    cout << "\nTest Complete\n" << endl;
    return 0;
}
