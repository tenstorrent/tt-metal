#pragma once
#include "common/utils.hpp"

namespace tt
{
namespace test_utils
{
    template <typename T>
    void print_vec(const std::vector<T>& vec) {
        int idx = 0;
        for (int i = 0; i < vec.size(); i++) {
            std::cout << vec.at(i) << ", " ;
        }
        std::cout << std::endl;
    }
}
}
