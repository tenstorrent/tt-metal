// ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp

template <typename T>
T UnaryDeviceOperation<T, T>::operator()(T x) const {
    return x * (static_cast<T>(3.14159265358979323846f) / 180);
}

template <>
float UnaryDeviceOperation<float, float>::operator()(float x) const {
    return x * (M_PI_F / 180);
}