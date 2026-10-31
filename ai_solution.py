```cpp
namespace ttnn {
namespace kernels {
namespace generic {

template<typename T, typename Context>
struct binary_composite_op<op::div_no_nan, T, Context> {
  using Output = T;
  template<typename SourceT>
  Output operator()(Context const& ctx, Inputs const& inputs) {
    auto a = inputs[0];
    auto b = inputs[1];
    if (b.isZero()) {
      return Output{0};
    }
    return a * (1.0 / b);
  }
};

} // namespace generic
} // namespace kernels
} // namespace ttnn
```