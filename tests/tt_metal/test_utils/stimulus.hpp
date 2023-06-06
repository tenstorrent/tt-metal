#include "common/utils.hpp"

#include <algorithm>
#include <random>


namespace tt
{
namespace test_utils
{
    template<typename T>
    std::vector<T> generate_uniform_int_random_vector (T min, T max, size_t size_in_bytes) {
        std::random_device rd;
        std::mt19937 gen(0);
        const size_t numel = size_in_bytes/sizeof(T);
        std::vector<T> values (numel);
        std::uniform_int_distribution<T> dis(min, max);
        std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
        return values;
    }

}
}
