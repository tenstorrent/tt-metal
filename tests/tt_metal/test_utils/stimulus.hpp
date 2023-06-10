#pragma once
#include "common/utils.hpp"

#include <algorithm>
#include <random>


namespace tt
{
namespace test_utils
{
    template<typename T>
    std::vector<T> generate_uniform_random_vector (T min, T max, const size_t numel, const float seed=0) {
        std::random_device rd;
        std::mt19937 gen(seed);
        std::vector<T> values (numel);
        if constexpr (std::is_integral<T>::value) {
            std::uniform_int_distribution<T> dis(min, max);
            std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
        } else {
            std::uniform_real_distribution<T> dis(min, max);
            std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
        }
        return values;
    }

    template<typename T>
    std::vector<T> generate_normal_random_vector (T mean, T std, const size_t numel, const float seed=0) {
        std::random_device rd;
        std::mt19937 gen(seed);
        std::vector<T> values (numel);
        std::normal_distribution<T> dis(mean, std);
        std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
        return values;
    }

}
}
