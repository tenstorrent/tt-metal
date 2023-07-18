namespace tt {

namespace tt_metal {

template <typename T>
bool is_power_of_two(T val) {
    return (val & (val-1))==T(0);
}

}  // namespace metal

}  // namespace tt
