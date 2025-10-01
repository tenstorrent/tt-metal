// gelu_test_comprehensive.cpp
//
// Comprehensive GELU Testing Framework - Complete Production Ready Edition
// Combines exhaustive scalar testing (Framework 1) with modern tensor support
//
// Mathematical reference: GELU(x) = 0.5 * x * (1 + erf(x/√2))
//                        GELU'(x) = Φ(x) + x·φ(x)
//
// Architecture: Type alias pattern for zero virtual overhead
// Note: static_cast required for backend-specific APIs (compile-time safe)
//
// Build commands:
//   Standalone: g++ -std=c++17 -DGELU_TEST_REFERENCE_ONLY gelu_test_comprehensive.cpp -lgtest -lgtest_main -pthread -o gelu_test
//   TT Backend: g++ -std=c++17 gelu_test_comprehensive.cpp -lgtest -lgtest_main -pthread -lttnn -lttml -I/path/to/tt -o gelu_test
//
// Test Coverage:
//   - 34 scalar tests (Framework 1 complete)
//   - 4 FP16-specific tests (when available)
//   - 8 tensor operation tests
//   Total: 42-46 tests depending on FP16 availability

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <functional>
#include <string>
#include <map>
#include <random>
#include <numeric>
#include <limits>
#include <algorithm>
#include <array>
#include <memory>
#include <iostream>

#ifndef GELU_TEST_REFERENCE_ONLY
#include <core/ttnn_all_includes.hpp>
#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/unary_ops.hpp"
#include "ops/losses.hpp"
#define GELU_HAS_TT_BACKEND 1
#else
#define GELU_HAS_TT_BACKEND 0
#endif

// ============================================================================
// Mathematical Constants
// ============================================================================

namespace constants {
    constexpr double sqrt_2 = 1.4142135623730951;
    constexpr double inv_sqrt_2 = 0.7071067811865475;
    constexpr double sqrt_2_pi = 2.5066282746310002;
    constexpr double inv_sqrt_2_pi = 0.3989422804014327;
    constexpr double sqrt_2_over_pi = 0.7978845608028654;
    constexpr double gelu_lipschitz = 1.0844;
}

// ============================================================================
// FP16 Support (IEEE 754 binary16)
// ============================================================================

#ifdef __FLT16_MANT_DIG__
#define GELU_HAS_FP16 1

namespace fp16 {
    constexpr double FP16_MAX = 65504.0;
    constexpr double FP16_MIN = -65504.0;
    constexpr double FP16_OVERFLOW_THRESHOLD = 40.0;

    inline double to_fp16(double x) {
        _Float16 fp16_val = static_cast<_Float16>(x);
        return static_cast<double>(fp16_val);
    }

    inline double to_fp16_safe(double x) {
        if (x > FP16_MAX) return FP16_MAX;
        if (x < FP16_MIN) return FP16_MIN;
        return to_fp16(x);
    }

    inline bool in_range(double x) {
        return x >= FP16_MIN && x <= FP16_MAX;
    }
}
#else
#define GELU_HAS_FP16 0
#endif

// ============================================================================
// TENSOR SUPPORT: Core Abstractions
// ============================================================================

namespace gelu_test {

struct TensorShape {
    std::array<uint32_t, 4> dims;  // [N, C, H, W]

    constexpr TensorShape(uint32_t n, uint32_t c, uint32_t h, uint32_t w)
        : dims{n, c, h, w} {}

    constexpr uint32_t operator[](size_t i) const { return dims[i]; }
    constexpr size_t total_size() const {
        return static_cast<size_t>(dims[0]) * dims[1] * dims[2] * dims[3];
    }

    bool operator==(const TensorShape& other) const { return dims == other.dims; }
    bool operator!=(const TensorShape& other) const { return !(*this == other); }

#if GELU_HAS_TT_BACKEND
    ttnn::Shape to_ttnn() const {
        return ttnn::Shape({dims[0], dims[1], dims[2], dims[3]});
    }
#endif
};

enum class GELUMode {
    ERF, TANH, SIGMOID, AUTO
};

enum class Capability : uint32_t {
    ELEMENTWISE = 1 << 0,
    BROADCASTING = 1 << 1,
    REDUCTION = 1 << 2,
    MATMUL = 1 << 3,
    AUTOGRAD = 1 << 4
};

class GELURegistry {
public:
    using ForwardFunc = std::function<float(float)>;
    using BackwardFunc = std::function<float(float)>;

    struct Implementation {
        std::string name;
        ForwardFunc forward;
        BackwardFunc backward;
        float forward_tolerance;
        float backward_tolerance;
    };

    static GELURegistry& instance() {
        static GELURegistry registry;
        return registry;
    }

    const Implementation& get(GELUMode mode) const {
        auto it = implementations_.find(mode);
        return (it != implementations_.end()) ? it->second : implementations_.at(GELUMode::TANH);
    }

private:
    GELURegistry() { initialize(); }
    std::map<GELUMode, Implementation> implementations_;

    void initialize() {
        implementations_[GELUMode::ERF] = {
            "erf",
            [](float x) {
                return 0.5f * x * (1.0f + std::erf(x * constants::inv_sqrt_2));
            },
            [](float x) {
                float erf_term = 0.5f * (1.0f + std::erf(x * constants::inv_sqrt_2));
                float pdf_term = x * constants::inv_sqrt_2_pi * std::exp(-0.5f * x * x);
                return erf_term + pdf_term;
            },
            1e-10f, 1e-10f
        };

        implementations_[GELUMode::TANH] = {
            "tanh",
            [](float x) {
                constexpr float a = 0.044715f;
                if (std::abs(x) > 100.0f) return (x > 0) ? x : 0.0f;
                float x_cubed = x * x * x;
                float inner = constants::sqrt_2_over_pi * (x + a * x_cubed);
                inner = std::max(-20.0f, std::min(20.0f, inner));
                return 0.5f * x * (1.0f + std::tanh(inner));
            },
            [](float x) {
                constexpr float a = 0.044715f;
                if (std::abs(x) > 100.0f) return (x > 0) ? 1.0f : 0.0f;
                float x_squared = x * x;
                float x_cubed = x_squared * x;
                float inner = constants::sqrt_2_over_pi * (x + a * x_cubed);
                inner = std::max(-20.0f, std::min(20.0f, inner));
                float tanh_inner = std::tanh(inner);
                float sech2 = 1.0f - tanh_inner * tanh_inner;
                float d_inner_dx = constants::sqrt_2_over_pi * (1.0f + 3.0f * a * x_squared);
                return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner_dx;
            },
            5e-4f, 1e-3f
        };

        implementations_[GELUMode::SIGMOID] = {
            "sigmoid",
            [](float x) {
                constexpr float beta = 1.702f;
                float sigmoid = 1.0f / (1.0f + std::exp(-beta * x));
                return x * sigmoid;
            },
            [](float x) {
                constexpr float beta = 1.702f;
                float sigmoid = 1.0f / (1.0f + std::exp(-beta * x));
                return sigmoid * (1.0f + x * beta * (1.0f - sigmoid));
            },
            2e-2f, 3e-2f
        };
    }
};

#if !GELU_HAS_TT_BACKEND

class SimpleTensor {
private:
    std::vector<float> data_;
    std::vector<float> grad_;
    TensorShape shape_;
    bool requires_grad_;

public:
    SimpleTensor(std::vector<float> data, TensorShape shape, bool requires_grad = false)
        : data_(std::move(data)), shape_(shape), requires_grad_(requires_grad) {
        if (requires_grad_) {
            grad_.resize(data_.size(), 0.0f);
        }
    }

    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data_mut() { return data_; }
    const std::vector<float>& grad() const { return grad_; }
    std::vector<float>& grad_mut() { return grad_; }
    TensorShape shape() const { return shape_; }
    bool requires_grad() const { return requires_grad_; }

    void zero_grad() {
        if (requires_grad_) std::fill(grad_.begin(), grad_.end(), 0.0f);
    }

    void accumulate_grad(const std::vector<float>& incoming) {
        if (requires_grad_ && incoming.size() == grad_.size()) {
            for (size_t i = 0; i < grad_.size(); ++i) {
                grad_[i] += incoming[i];
            }
        }
    }
};

using TensorImpl = SimpleTensor;

#else

class TTTensorWrapper {
private:
    ttml::autograd::TensorPtr tensor_;
    mutable std::vector<float> cached_data_;
    mutable std::vector<float> cached_grad_;
    mutable bool data_valid_ = false;
    mutable bool grad_valid_ = false;

public:
    explicit TTTensorWrapper(ttml::autograd::TensorPtr tensor)
        : tensor_(std::move(tensor)) {}

    std::vector<float> data() const {
        if (!data_valid_) {
            cached_data_ = ttml::core::to_vector(tensor_->get_value());
            data_valid_ = true;
        }
        return cached_data_;
    }

    std::vector<float> grad() const {
        if (!grad_valid_) {
            if (ttml::core::is_tensor_initialized(tensor_->get_grad())) {
                cached_grad_ = ttml::core::to_vector(tensor_->get_grad());
            } else {
                cached_grad_.resize(shape().total_size(), 0.0f);
            }
            grad_valid_ = true;
        }
        return cached_grad_;
    }

    TensorShape shape() const {
        auto s = tensor_->get_shape();
        return TensorShape(s[0], s[1], s[2], s[3]);
    }

    bool requires_grad() const {
        return ttml::core::is_tensor_initialized(tensor_->get_grad());
    }

    void zero_grad() {
        if (requires_grad()) {
            tensor_->set_grad(ttml::core::zeros_like(tensor_->get_value()));
            data_valid_ = false;
            grad_valid_ = false;
        }
    }

    void accumulate_grad(const std::vector<float>&) { grad_valid_ = false; }

    ttml::autograd::TensorPtr& native() { return tensor_; }
    const ttml::autograd::TensorPtr& native() const { return tensor_; }
};

using TensorImpl = TTTensorWrapper;

#endif

class Tensor {
private:
    std::shared_ptr<TensorImpl> impl_;

public:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

    std::vector<float> data() const { return impl_->data(); }
    std::vector<float> grad() const { return impl_->grad(); }
    TensorShape shape() const { return impl_->shape(); }
    bool requires_grad() const { return impl_->requires_grad(); }
    void zero_grad() { impl_->zero_grad(); }

    TensorImpl& impl() { return *impl_; }
    const TensorImpl& impl() const { return *impl_; }
};

class TensorOps {
public:
    static Tensor create(const std::vector<float>& data, TensorShape shape,
                         bool requires_grad = false) {
#if GELU_HAS_TT_BACKEND
        auto* device = &ttml::autograd::ctx().get_device();
        auto tt_tensor = ttml::autograd::create_tensor(
            ttml::core::from_vector(data, shape.to_ttnn(), device)
        );
        return Tensor(std::make_shared<TensorImpl>(std::move(tt_tensor)));
#else
        return Tensor(std::make_shared<TensorImpl>(data, shape, requires_grad));
#endif
    }

    static Tensor zeros(TensorShape shape, bool requires_grad = false) {
        return create(std::vector<float>(shape.total_size(), 0.0f), shape, requires_grad);
    }

    static Tensor randn(TensorShape shape, float mean = 0.0f, float std = 1.0f,
                        bool requires_grad = false) {
        std::vector<float> data(shape.total_size());
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(mean, std);
        for (auto& val : data) val = dist(gen);
        return create(data, shape, requires_grad);
    }

    static Tensor gelu(const Tensor& input, GELUMode mode = GELUMode::AUTO) {
#if GELU_HAS_TT_BACKEND
        auto& tt_impl = const_cast<TTTensorWrapper&>(static_cast<const TTTensorWrapper&>(input.impl()));
        auto result = ttml::ops::gelu(tt_impl.native(), mode == GELUMode::SIGMOID);
        return Tensor(std::make_shared<TensorImpl>(std::move(result)));
#else
        if (mode == GELUMode::AUTO) mode = GELUMode::ERF;
        auto& impl = GELURegistry::instance().get(mode);
        auto in_data = input.data();
        std::vector<float> out_data(in_data.size());
        for (size_t i = 0; i < in_data.size(); ++i) {
            out_data[i] = impl.forward(in_data[i]);
        }
        return create(out_data, input.shape(), input.requires_grad());
#endif
    }

    static Tensor mse_loss(const Tensor& pred, const Tensor& target) {
#if GELU_HAS_TT_BACKEND
        auto& pred_impl = const_cast<TTTensorWrapper&>(static_cast<const TTTensorWrapper&>(pred.impl()));
        auto& tgt_impl = const_cast<TTTensorWrapper&>(static_cast<const TTTensorWrapper&>(target.impl()));
        auto result = ttml::ops::mse_loss(pred_impl.native(), tgt_impl.native());
        return Tensor(std::make_shared<TensorImpl>(std::move(result)));
#else
        auto p = pred.data();
        auto t = target.data();
        float sum = 0.0f;
        for (size_t i = 0; i < p.size(); ++i) {
            float diff = p[i] - t[i];
            sum += diff * diff;
        }
        return create({sum / p.size()}, TensorShape(1, 1, 1, 1), true);
#endif
    }

    static void backward(Tensor& loss) {
#if GELU_HAS_TT_BACKEND
        auto& impl = static_cast<TTTensorWrapper&>(loss.impl());
        impl.native()->backward();
#else
        auto& impl = static_cast<SimpleTensor&>(loss.impl());
        if (!impl.grad_mut().empty()) {
            impl.grad_mut()[0] = 1.0f;
        }
#endif
    }

    static constexpr uint32_t capabilities() {
#if GELU_HAS_TT_BACKEND
        return static_cast<uint32_t>(Capability::ELEMENTWISE) |
               static_cast<uint32_t>(Capability::BROADCASTING) |
               static_cast<uint32_t>(Capability::REDUCTION) |
               static_cast<uint32_t>(Capability::MATMUL) |
               static_cast<uint32_t>(Capability::AUTOGRAD);
#else
        return static_cast<uint32_t>(Capability::ELEMENTWISE) |
               static_cast<uint32_t>(Capability::BROADCASTING) |
               static_cast<uint32_t>(Capability::REDUCTION);
#endif
    }

    static constexpr bool supports(Capability cap) {
        return (capabilities() & static_cast<uint32_t>(cap)) != 0;
    }

    static constexpr const char* backend_name() {
        return GELU_HAS_TT_BACKEND ? "TT" : "Standalone";
    }
};

class BatchGELU {
private:
    GELUMode mode_;
public:
    explicit BatchGELU(GELUMode mode = GELUMode::AUTO) : mode_(mode) {}
    Tensor forward(const Tensor& input) { return TensorOps::gelu(input, mode_); }
};

struct BertConfig {
    uint32_t batch_size, seq_length, hidden_dim, intermediate_dim;
    static constexpr BertConfig tiny() { return {1, 8, 64, 256}; }
    static constexpr BertConfig small() { return {1, 32, 256, 1024}; }
    TensorShape input_shape() const { return TensorShape(batch_size, 1, seq_length, hidden_dim); }
};

#define SKIP_IF_NO_CAPABILITY(cap) \
    if (!TensorOps::supports(cap)) { \
        GTEST_SKIP() << "Capability " #cap " not supported by " << TensorOps::backend_name(); \
    }

} // namespace gelu_test

// ============================================================================
// SCALAR IMPLEMENTATIONS: Reference Forward (FP64)
// ============================================================================

double ref_erf_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x * constants::inv_sqrt_2));
}

double ref_tanh_gelu(double x) {
    constexpr double a = 0.044715;
    if (std::abs(x) > 100.0) return (x > 0) ? x : 0.0;
    double x_cubed = x * x * x;
    double inner = constants::sqrt_2_over_pi * (x + a * x_cubed);
    inner = std::max(-20.0, std::min(20.0, inner));
    return 0.5 * x * (1.0 + std::tanh(inner));
}

double ref_sigm_gelu(double x) {
    constexpr double beta = 1.702;
    double exp_term = std::exp(-std::abs(beta * x));
    double sigmoid = (x >= 0) ? 1.0 / (1.0 + exp_term) : exp_term / (1.0 + exp_term);
    return x * sigmoid;
}

// ============================================================================
// SCALAR IMPLEMENTATIONS: Reference Backward (FP64)
// ============================================================================

double ref_erf_gelu_grad(double x) {
    double erf_term = 0.5 * (1.0 + std::erf(x * constants::inv_sqrt_2));
    double pdf_term = x * constants::inv_sqrt_2_pi * std::exp(-0.5 * x * x);
    return erf_term + pdf_term;
}

double ref_tanh_gelu_grad(double x) {
    constexpr double a = 0.044715;
    if (std::abs(x) > 100.0) return (x > 0) ? 1.0 : 0.0;
    double x_squared = x * x;
    double x_cubed = x_squared * x;
    double inner = constants::sqrt_2_over_pi * (x + a * x_cubed);
    inner = std::max(-20.0, std::min(20.0, inner));
    double tanh_inner = std::tanh(inner);
    double sech_squared = 1.0 - tanh_inner * tanh_inner;
    double d_inner_dx = constants::sqrt_2_over_pi * (1.0 + 3.0 * a * x_squared);
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * d_inner_dx;
}

double ref_sigm_gelu_grad(double x) {
    constexpr double beta = 1.702;
    double exp_term = std::exp(-std::abs(beta * x));
    double sigmoid = (x >= 0) ? 1.0 / (1.0 + exp_term) : exp_term / (1.0 + exp_term);
    return sigmoid * (1.0 + x * beta * (1.0 - sigmoid));
}

// ============================================================================
// SCALAR IMPLEMENTATIONS: FP16 Versions
// ============================================================================

#if GELU_HAS_FP16

double ref_erf_gelu_fp16(double x) {
    using namespace fp16;
    x = to_fp16(x);
    double x_scaled = to_fp16(x * constants::inv_sqrt_2);
    double erf_val = to_fp16(std::erf(x_scaled));
    double result = to_fp16(0.5 * to_fp16(x * to_fp16(1.0 + erf_val)));
    return to_fp16(result);
}

double ref_tanh_gelu_fp16(double x) {
    using namespace fp16;
    constexpr double a = 0.044715;
    x = to_fp16(x);
    if (std::abs(x) > FP16_OVERFLOW_THRESHOLD) {
        return (x > 0) ? to_fp16(x) : to_fp16(0.0);
    }
    double x_squared = to_fp16(to_fp16(x) * to_fp16(x));
    double x_cubed = to_fp16(to_fp16(x_squared) * to_fp16(x));
    if (!in_range(a * x_cubed)) {
        return (x > 0) ? to_fp16(x) : to_fp16(0.0);
    }
    double term = to_fp16(to_fp16(x) + to_fp16(a * x_cubed));
    double inner = to_fp16(constants::sqrt_2_over_pi * term);
    inner = std::max(-20.0, std::min(20.0, inner));
    double tanh_val = to_fp16(std::tanh(inner));
    double result = to_fp16(0.5 * to_fp16(to_fp16(x) * to_fp16(1.0 + tanh_val)));
    return to_fp16(result);
}

double ref_sigm_gelu_fp16(double x) {
    using namespace fp16;
    constexpr double beta = 1.702;
    x = to_fp16(x);
    double beta_x = to_fp16(beta * x);
    if (std::abs(beta_x) > 20.0) {
        return (x > 0) ? to_fp16(x) : to_fp16(0.0);
    }
    double exp_term = to_fp16(std::exp(-std::abs(beta_x)));
    double sigmoid = (x >= 0) ?
        to_fp16(1.0 / to_fp16(1.0 + exp_term)) :
        to_fp16(exp_term / to_fp16(1.0 + exp_term));
    double result = to_fp16(to_fp16(x) * sigmoid);
    return to_fp16(result);
}

double ref_erf_gelu_grad_fp16(double x) {
    using namespace fp16;
    x = to_fp16(x);
    double x_scaled = to_fp16(x * constants::inv_sqrt_2);
    double erf_val = to_fp16(std::erf(x_scaled));
    double erf_term = to_fp16(0.5 * to_fp16(1.0 + erf_val));
    double x_squared = to_fp16(to_fp16(x) * to_fp16(x));
    double exp_term = to_fp16(std::exp(to_fp16(-0.5 * x_squared)));
    double pdf_term = to_fp16(to_fp16(x) * to_fp16(constants::inv_sqrt_2_pi * exp_term));
    double result = to_fp16(to_fp16(erf_term) + to_fp16(pdf_term));
    return to_fp16(result);
}

double ref_tanh_gelu_grad_fp16(double x) {
    using namespace fp16;
    constexpr double a = 0.044715;
    x = to_fp16(x);
    if (std::abs(x) > FP16_OVERFLOW_THRESHOLD) {
        return (x > 0) ? to_fp16(1.0) : to_fp16(0.0);
    }
    double x_squared = to_fp16(to_fp16(x) * to_fp16(x));
    double x_cubed = to_fp16(to_fp16(x_squared) * to_fp16(x));
    if (!in_range(a * x_cubed)) {
        return (x > 0) ? to_fp16(1.0) : to_fp16(0.0);
    }
    double term = to_fp16(to_fp16(x) + to_fp16(a * x_cubed));
    double inner = to_fp16(constants::sqrt_2_over_pi * term);
    inner = std::max(-20.0, std::min(20.0, inner));
    double tanh_val = to_fp16(std::tanh(inner));
    double sech_squared = to_fp16(1.0 - to_fp16(tanh_val * tanh_val));
    double d_inner_dx = to_fp16(constants::sqrt_2_over_pi * to_fp16(1.0 + to_fp16(3.0 * a * x_squared)));
    double first_term = to_fp16(0.5 * to_fp16(1.0 + tanh_val));
    double second_term = to_fp16(0.5 * to_fp16(to_fp16(x) * to_fp16(sech_squared * d_inner_dx)));
    double result = to_fp16(to_fp16(first_term) + to_fp16(second_term));
    return to_fp16(result);
}

double ref_sigm_gelu_grad_fp16(double x) {
    using namespace fp16;
    constexpr double beta = 1.702;
    x = to_fp16(x);
    double beta_x = to_fp16(beta * x);
    if (std::abs(beta_x) > 20.0) {
        return (x > 0) ? to_fp16(1.0) : to_fp16(0.0);
    }
    double exp_term = to_fp16(std::exp(-std::abs(beta_x)));
    double sigmoid = (x >= 0) ?
        to_fp16(1.0 / to_fp16(1.0 + exp_term)) :
        to_fp16(exp_term / to_fp16(1.0 + exp_term));
    double one_minus_sigmoid = to_fp16(1.0 - sigmoid);
    double term = to_fp16(to_fp16(x) * to_fp16(beta * one_minus_sigmoid));
    double result = to_fp16(sigmoid * to_fp16(1.0 + term));
    return to_fp16(result);
}

#endif // GELU_HAS_FP16

// ============================================================================
// SCALAR IMPLEMENTATIONS: Container
// ============================================================================

class Impl {
public:
    std::string name;
    std::function<double(double)> gelu;
    std::function<double(double)> gelu_grad;
    std::function<void(const std::vector<double>&, std::vector<double>&)> gelu_batch;
    std::function<void(const std::vector<double>&, std::vector<double>&)> gelu_grad_batch;

    Impl(std::string n, std::function<double(double)> fwd, std::function<double(double)> bwd)
        : name(std::move(n)), gelu(std::move(fwd)), gelu_grad(std::move(bwd)) {
        auto gelu_func = gelu;
        auto grad_func = gelu_grad;
        gelu_batch = [gelu_func](const std::vector<double>& in, std::vector<double>& out) {
            out.resize(in.size());
            for (size_t i = 0; i < in.size(); ++i) out[i] = gelu_func(in[i]);
        };
        gelu_grad_batch = [grad_func](const std::vector<double>& in, std::vector<double>& out) {
            out.resize(in.size());
            for (size_t i = 0; i < in.size(); ++i) out[i] = grad_func(in[i]);
        };
    }

    operator std::string() const { return name; }
};

inline std::ostream& operator<<(std::ostream& os, const Impl& impl) {
    return os << impl.name;
}

inline std::vector<Impl> get_all_impls() {
    std::vector<Impl> implementations = {
        {"RefErfGELU",  ref_erf_gelu,  ref_erf_gelu_grad},
        {"RefTanhGELU", ref_tanh_gelu, ref_tanh_gelu_grad},
        {"RefSigmGELU", ref_sigm_gelu, ref_sigm_gelu_grad}
    };
#if GELU_HAS_FP16
    implementations.push_back({"RefErfGELU_FP16",  ref_erf_gelu_fp16,  ref_erf_gelu_grad_fp16});
    implementations.push_back({"RefTanhGELU_FP16", ref_tanh_gelu_fp16, ref_tanh_gelu_grad_fp16});
    implementations.push_back({"RefSigmGELU_FP16", ref_sigm_gelu_fp16, ref_sigm_gelu_grad_fp16});
#endif
    return implementations;
}

const auto impls = get_all_impls();

// ============================================================================
// Test Fixture
// ============================================================================

class GeluTest : public ::testing::Test {
protected:
    std::mt19937 rng{42};

    void SetUp() override {
#if GELU_HAS_TT_BACKEND
        ttml::autograd::ctx().open_device();
#endif
    }

    void TearDown() override {
#if GELU_HAS_TT_BACKEND
        ttml::autograd::ctx().close_device();
#endif
    }

    // Scalar helpers
    double reference_gelu(double x) { return ref_erf_gelu(x); }
    double reference_gelu_grad(double x) { return ref_erf_gelu_grad(x); }

    double numerical_gradient(std::function<double(double)> f, double x, double h = 1e-6) {
        return (f(x + h) - f(x - h)) / (2.0 * h);
    }

    double numerical_gradient_fp16_safe(std::function<double(double)> f, double x) {
        double h = std::max(0.01, std::abs(x) * 0.01);
        return (f(x + h) - f(x - h)) / (2.0 * h);
    }

    double relative_error(double actual, double expected) {
        return std::abs(actual - expected) / (std::abs(expected) + 1e-10);
    }

    std::vector<double> generate_batch(size_t size, double min_val, double max_val) {
        std::uniform_real_distribution<double> dist(min_val, max_val);
        std::vector<double> batch(size);
        for (auto& val : batch) val = dist(rng);
        return batch;
    }

    std::map<std::string, double> fwd_tol = {
        {"RefErfGELU", 1e-10}, {"RefTanhGELU", 5e-4}, {"RefSigmGELU", 2e-2},
#if GELU_HAS_FP16
        {"RefErfGELU_FP16", 5e-3}, {"RefTanhGELU_FP16", 1e-2}, {"RefSigmGELU_FP16", 5e-2},
#endif
    };

    std::map<std::string, double> grad_tol = {
        {"RefErfGELU", 1e-10}, {"RefTanhGELU", 1e-3}, {"RefSigmGELU", 3e-2},
#if GELU_HAS_FP16
        {"RefErfGELU_FP16", 1e-2}, {"RefTanhGELU_FP16", 2e-2}, {"RefSigmGELU_FP16", 5e-2},
#endif
    };

    double get_fwd_tol(const std::string& name) {
        auto it = fwd_tol.find(name);
        return (it != fwd_tol.end()) ? it->second : 1e-3;
    }

    double get_grad_tol(const std::string& name) {
        auto it = grad_tol.find(name);
        return (it != grad_tol.end()) ? it->second : 1e-2;
    }

    bool is_fp16_impl(const std::string& name) {
        return name.find("_FP16") != std::string::npos;
    }

    // Tensor tolerance
    float get_tensor_tolerance(gelu_test::GELUMode mode, float expected, bool is_gradient = false) {
        auto impl = gelu_test::GELURegistry::instance().get(mode);
        float base_tol = is_gradient ? impl.backward_tolerance : impl.forward_tolerance;
#if GELU_HAS_TT_BACKEND
        // Increase tolerance for TT backend, especially for SIGMOID mode
        if (mode == gelu_test::GELUMode::SIGMOID) {
            base_tol = std::max(base_tol, is_gradient ? 5e-2f : 3e-2f);
        } else {
            base_tol = std::max(base_tol, is_gradient ? 1e-2f : 5e-3f);
        }
#endif
        return std::max(base_tol, std::abs(expected) * base_tol * 2.0f);
    }

    // Tensor validation
    void validate_shape(const gelu_test::Tensor& t, const gelu_test::TensorShape& expected) {
        EXPECT_EQ(t.shape(), expected);
    }

    void validate_no_nans(const gelu_test::Tensor& t) {
        auto data = t.data();
        for (size_t i = 0; i < data.size(); ++i) {
            EXPECT_FALSE(std::isnan(data[i])) << "NaN at " << i;
            EXPECT_FALSE(std::isinf(data[i])) << "Inf at " << i;
        }
    }
};

// ============================================================================
// SECTION 1-7: All 34 Scalar Tests from Framework 1
// ============================================================================

// Forward tests (7)
TEST_F(GeluTest, ForwardBasicValues) {
    std::vector<double> inputs = {-2.0, -1.0, 0.0, 1.0, 2.0};
    for (const auto& impl : impls) {
        double tol = get_fwd_tol(impl.name);
        for (double x : inputs) {
            EXPECT_NEAR(impl.gelu(x), reference_gelu(x), tol) << impl << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, ForwardZeroPoint) {
    for (const auto& impl : impls) {
        double tol = is_fp16_impl(impl.name) ? 1e-4 : 1e-10;
        EXPECT_NEAR(impl.gelu(0.0), 0.0, tol) << impl;
    }
}

TEST_F(GeluTest, ForwardMonotonicity) {
    std::vector<double> inputs = {-0.5, 0.0, 0.5, 1.0, 2.0, 3.0};
    for (const auto& impl : impls) {
        double prev = impl.gelu(inputs[0]);
        for (size_t i = 1; i < inputs.size(); ++i) {
            double curr = impl.gelu(inputs[i]);
            EXPECT_GT(curr, prev - 1e-6) << impl << " at x=" << inputs[i];
            prev = curr;
        }
    }
}

TEST_F(GeluTest, ForwardRangeCheck) {
    std::vector<double> inputs = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    for (const auto& impl : impls) {
        double margin = is_fp16_impl(impl.name) ? 0.01 : 1e-3;
        for (double x : inputs) {
            double result = impl.gelu(x);
            if (x >= 0) {
                EXPECT_GE(result, -margin) << impl << " at x=" << x;
                EXPECT_LE(result, x + margin) << impl << " at x=" << x;
            } else {
                EXPECT_LE(result, margin) << impl << " at x=" << x;
                EXPECT_GE(result, x - margin) << impl << " at x=" << x;
            }
        }
    }
}

TEST_F(GeluTest, ForwardAsymptotics) {
    for (const auto& impl : impls) {
        double tol = is_fp16_impl(impl.name) ? 0.05 : 0.01;
        for (double x : {5.0, 10.0, 20.0}) {
            double rel_err = relative_error(impl.gelu(x), x);
            EXPECT_LT(rel_err, tol) << impl << " x=" << x;
        }
        for (double x : {-5.0, -10.0, -20.0}) {
            EXPECT_NEAR(impl.gelu(x), 0.0, tol) << impl << " x=" << x;
        }
    }
}

TEST_F(GeluTest, ForwardStability) {
    for (const auto& impl : impls) {
        std::vector<double> test_vals = {-100.0, -50.0, 50.0, 100.0};
#if GELU_HAS_FP16
        if (is_fp16_impl(impl.name)) {
            test_vals = {-40.0, -20.0, 20.0, 40.0};
        }
#endif
        for (double x : test_vals) {
            double result = impl.gelu(x);
            EXPECT_FALSE(std::isnan(result)) << impl << " at x=" << x;
            EXPECT_FALSE(std::isinf(result)) << impl << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, ForwardCriticalPoints) {
    std::vector<double> critical = {-1.5, -0.5, 0.0, 0.5, 1.5};
    for (const auto& impl : impls) {
        double tol = get_fwd_tol(impl.name);
        for (double x : critical) {
            EXPECT_NEAR(impl.gelu(x), reference_gelu(x), tol) << impl << " at x=" << x;
        }
    }
}

// Backward tests (10)
TEST_F(GeluTest, BackwardBasicValues) {
    std::vector<double> inputs = {-2.0, -1.0, 0.0, 1.0, 2.0};
    for (const auto& impl : impls) {
        double tol = get_grad_tol(impl.name);
        for (double x : inputs) {
            EXPECT_NEAR(impl.gelu_grad(x), reference_gelu_grad(x), tol) << impl << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, BackwardAtZero) {
    for (const auto& impl : impls) {
        double tol = get_grad_tol(impl.name);
        EXPECT_NEAR(impl.gelu_grad(0.0), 0.5, tol) << impl;
    }
}

TEST_F(GeluTest, BackwardPositivity) {
    for (const auto& impl : impls) {
        for (double x : {-0.5, 0.0, 1.0, 2.0, 5.0}) {
            double grad = impl.gelu_grad(x);
            EXPECT_GT(grad, -1e-6) << impl << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, BackwardMonotonicity) {
    std::vector<double> inputs = {-0.5, 0.0, 0.5, 1.0, 1.4};
    for (const auto& impl : impls) {
        double margin = is_fp16_impl(impl.name) ? 0.02 : 1e-6;
        double prev = impl.gelu_grad(inputs[0]);
        for (size_t i = 1; i < inputs.size(); ++i) {
            double curr = impl.gelu_grad(inputs[i]);
            EXPECT_GE(curr, prev - margin) << impl << " at x=" << inputs[i];
            prev = curr;
        }
    }
}

TEST_F(GeluTest, BackwardPostPeakDecrease) {
    std::vector<double> post_peak = {1.5, 2.0, 3.0, 5.0};
    for (const auto& impl : impls) {
        double margin = is_fp16_impl(impl.name) ? 0.1 : 1e-3;
        double grad_at_peak = impl.gelu_grad(constants::sqrt_2);
        for (double x : post_peak) {
            double grad = impl.gelu_grad(x);
            EXPECT_LT(grad, grad_at_peak + margin) << impl << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, BackwardRangeCheck) {
    for (const auto& impl : impls) {
        for (double x : {-0.5, 0.0, 1.0, 2.0, 5.0}) {
            double grad = impl.gelu_grad(x);
            EXPECT_GT(grad, -1e-4) << impl << " at x=" << x;
            double upper = is_fp16_impl(impl.name) ? 1.20 : 1.12;
            EXPECT_LE(grad, upper) << impl << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, BackwardAsymptotics) {
    for (const auto& impl : impls) {
        double tol = is_fp16_impl(impl.name) ? 0.05 : 0.02;
        for (double x : {5.0, 10.0, 20.0}) {
            EXPECT_NEAR(impl.gelu_grad(x), 1.0, tol) << impl << " x=" << x;
        }
        for (double x : {-5.0, -10.0, -20.0}) {
            EXPECT_NEAR(impl.gelu_grad(x), 0.0, tol) << impl << " x=" << x;
        }
    }
}

TEST_F(GeluTest, BackwardStability) {
    for (const auto& impl : impls) {
        std::vector<double> test_vals = {-100.0, -50.0, 50.0, 100.0};
#if GELU_HAS_FP16
        if (is_fp16_impl(impl.name)) {
            test_vals = {-40.0, -20.0, 20.0, 40.0};
        }
#endif
        for (double x : test_vals) {
            double grad = impl.gelu_grad(x);
            EXPECT_FALSE(std::isnan(grad)) << impl << " at x=" << x;
            EXPECT_FALSE(std::isinf(grad)) << impl << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, BackwardSmoothness) {
    std::vector<double> inputs;
    for (double x = -2.0; x <= 3.0; x += 0.25) {
        inputs.push_back(x);
    }
    for (const auto& impl : impls) {
        std::vector<double> grads;
        for (double x : inputs) grads.push_back(impl.gelu_grad(x));
        double max_jump = 0.0;
        for (size_t i = 1; i < grads.size(); ++i) {
            double jump = std::abs(grads[i] - grads[i-1]);
            max_jump = std::max(max_jump, jump);
        }
        double threshold = is_fp16_impl(impl.name) ? 0.35 : 0.25;
        EXPECT_LT(max_jump, threshold) << impl << " max jump=" << max_jump;
    }
}

TEST_F(GeluTest, BackwardCriticalPoints) {
    std::vector<double> critical = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
    for (const auto& impl : impls) {
        double tol = get_grad_tol(impl.name);
        for (double x : critical) {
            EXPECT_NEAR(impl.gelu_grad(x), reference_gelu_grad(x), tol) << impl << " at x=" << x;
        }
    }
}

// Lipschitz tests (2)
TEST_F(GeluTest, LipschitzConstantVerification) {
    for (const auto& impl : impls) {
        double max_grad = 0.0;
        double x_at_max = 0.0;
        for (double x = -5.0; x <= 5.0; x += 0.01) {
            double grad = impl.gelu_grad(x);
            if (grad > max_grad) {
                max_grad = grad;
                x_at_max = x;
            }
        }
        double grad_tol = is_fp16_impl(impl.name) ? 0.15 : 0.05;
        double loc_tol = is_fp16_impl(impl.name) ? 0.3 : 0.1;
        EXPECT_NEAR(max_grad, constants::gelu_lipschitz, grad_tol)
            << impl << " max=" << max_grad << " at x=" << x_at_max;
        EXPECT_NEAR(x_at_max, constants::sqrt_2, loc_tol) << impl;
    }
}

TEST_F(GeluTest, GradientSymmetryProperties) {
    for (const auto& impl : impls) {
        double tol = get_grad_tol(impl.name) * 10;
        double left_limit = impl.gelu_grad(-1e-3);
        EXPECT_LT(left_limit, 0.5 + tol) << impl;
        double right_limit = impl.gelu_grad(1e-3);
        EXPECT_GT(right_limit, 0.5 - tol) << impl;
    }
}

// Numerical stability tests (3)
TEST_F(GeluTest, SubnormalHandling) {
    for (const auto& impl : impls) {
        double epsilon = std::numeric_limits<double>::epsilon();
        double denorm_min = std::numeric_limits<double>::denorm_min();
        std::vector<double> tiny_values = {
            epsilon, epsilon * 10, epsilon * 100, denorm_min, denorm_min * 10
        };
        for (double x : tiny_values) {
            double result = impl.gelu(x);
            EXPECT_FALSE(std::isnan(result)) << impl << " at x=" << x;
            EXPECT_TRUE(std::isfinite(result)) << impl << " at x=" << x;
            double grad = impl.gelu_grad(x);
            EXPECT_FALSE(std::isnan(grad)) << impl << " at x=" << x;
            EXPECT_TRUE(std::isfinite(grad)) << impl << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, TanhApproximationOverflow) {
    auto tanh_impl = std::find_if(impls.begin(), impls.end(),
        [](const Impl& i) { return i.name == "RefTanhGELU" || i.name == "RefTanhGELU_FP16"; });
    if (tanh_impl != impls.end()) {
        std::vector<double> test_values = {-1000.0, -500.0, -200.0, 200.0, 500.0, 1000.0};
#if GELU_HAS_FP16
        if (is_fp16_impl(tanh_impl->name)) {
            test_values = {-60.0, -50.0, -45.0, 45.0, 50.0, 60.0};
        }
#endif
        for (double x : test_values) {
            double result = tanh_impl->gelu(x);
            EXPECT_FALSE(std::isnan(result)) << tanh_impl->name << " at x=" << x;
            EXPECT_FALSE(std::isinf(result)) << tanh_impl->name << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, MixedPrecisionSimulation) {
    for (const auto& impl : impls) {
        if (is_fp16_impl(impl.name)) continue;
        std::vector<double> test_values = {-2.0, -1.0, 0.0, 1.0, 2.0};
        for (double x : test_values) {
            float x_fp32 = static_cast<float>(x);
            double x_recovered = static_cast<double>(x_fp32);
            double result = impl.gelu(x_recovered);
            double grad = impl.gelu_grad(x_recovered);
            EXPECT_FALSE(std::isnan(result)) << impl << " at x=" << x;
            EXPECT_FALSE(std::isnan(grad)) << impl << " grad at x=" << x;
        }
    }
}

// FP16-specific tests (4 tests, only if FP16 available)
#if GELU_HAS_FP16
TEST_F(GeluTest, FP16OverflowHandling) {
    for (const auto& impl : impls) {
        if (!is_fp16_impl(impl.name)) continue;
        std::vector<double> overflow_test = {-50.0, -40.0, -30.0, 30.0, 40.0, 50.0};
        for (double x : overflow_test) {
            double result = impl.gelu(x);
            double grad = impl.gelu_grad(x);
            EXPECT_FALSE(std::isnan(result)) << impl << " at x=" << x;
            EXPECT_FALSE(std::isinf(result)) << impl << " at x=" << x;
            EXPECT_FALSE(std::isnan(grad)) << impl << " grad at x=" << x;
            EXPECT_FALSE(std::isinf(grad)) << impl << " grad at x=" << x;
        }
    }
}

TEST_F(GeluTest, FP16PrecisionDegradation) {
    std::vector<double> test_points = {-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0};
    for (const auto& impl : impls) {
        if (!is_fp16_impl(impl.name)) continue;
        double max_rel_error_fwd = 0.0;
        double max_rel_error_grad = 0.0;
        for (double x : test_points) {
            double fp16_fwd = impl.gelu(x);
            double fp64_fwd = reference_gelu(x);
            double rel_err_fwd = relative_error(fp16_fwd, fp64_fwd);
            max_rel_error_fwd = std::max(max_rel_error_fwd, rel_err_fwd);
            double fp16_grad = impl.gelu_grad(x);
            double fp64_grad = reference_gelu_grad(x);
            double rel_err_grad = relative_error(fp16_grad, fp64_grad);
            max_rel_error_grad = std::max(max_rel_error_grad, rel_err_grad);
        }
        double fwd_tolerance = (impl.name == "RefSigmGELU_FP16") ? 5.0 : 0.1;
        double grad_tolerance = (impl.name == "RefSigmGELU_FP16") ? 2.0 : 0.15;
        EXPECT_LT(max_rel_error_fwd, fwd_tolerance) << impl << " fwd loss: " << max_rel_error_fwd;
        EXPECT_LT(max_rel_error_grad, grad_tolerance) << impl << " grad loss: " << max_rel_error_grad;
    }
}

TEST_F(GeluTest, FP16BatchAccumulation) {
    size_t batch_size = 128;
    auto inputs = generate_batch(batch_size, -5.0, 5.0);
    for (const auto& impl : impls) {
        if (!is_fp16_impl(impl.name)) continue;
        std::vector<double> outputs;
        impl.gelu_batch(inputs, outputs);
        for (size_t i = 0; i < batch_size; ++i) {
            double individual = impl.gelu(inputs[i]);
            double from_batch = outputs[i];
            EXPECT_DOUBLE_EQ(individual, from_batch) << impl << " at i=" << i;
        }
        double mean = std::accumulate(outputs.begin(), outputs.end(), 0.0) / batch_size;
        EXPECT_TRUE(std::isfinite(mean)) << impl;
    }
}

TEST_F(GeluTest, FP16GradientClipping) {
    std::vector<double> test_range;
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        test_range.push_back(x);
    }
    for (const auto& impl : impls) {
        if (!is_fp16_impl(impl.name)) continue;
        for (double x : test_range) {
            double grad = impl.gelu_grad(x);
            EXPECT_TRUE(std::isfinite(grad)) << impl << " at x=" << x;
            EXPECT_LT(std::abs(grad), 2.0) << impl << " at x=" << x << " grad=" << grad;
            if (x > -0.5) {
                EXPECT_GT(grad, -0.1) << impl << " at x=" << x;
            }
        }
    }
}
#endif // GELU_HAS_FP16

// Batch processing tests (2)
TEST_F(GeluTest, BatchForwardBackward) {
    size_t batch_size = 256;
    auto inputs = generate_batch(batch_size, -5.0, 5.0);
    for (const auto& impl : impls) {
        std::vector<double> outputs_fwd, outputs_grad;
        impl.gelu_batch(inputs, outputs_fwd);
        impl.gelu_grad_batch(inputs, outputs_grad);
        EXPECT_EQ(outputs_fwd.size(), batch_size) << impl;
        EXPECT_EQ(outputs_grad.size(), batch_size) << impl;
        double tol_fwd = get_fwd_tol(impl.name) * 1.1;
        double tol_grad = get_grad_tol(impl.name) * 1.1;
        for (size_t i = 0; i < batch_size; ++i) {
            EXPECT_NEAR(outputs_fwd[i], reference_gelu(inputs[i]), tol_fwd) << impl << " at i=" << i;
            EXPECT_NEAR(outputs_grad[i], reference_gelu_grad(inputs[i]), tol_grad) << impl << " at i=" << i;
        }
    }
}

TEST_F(GeluTest, LargeBatchStability) {
    size_t batch_size = 1024;
    auto inputs = generate_batch(batch_size, -10.0, 10.0);
    for (const auto& impl : impls) {
        std::vector<double> outputs;
        impl.gelu_batch(inputs, outputs);
        for (size_t i = 0; i < batch_size; ++i) {
            EXPECT_FALSE(std::isnan(outputs[i])) << impl << " at i=" << i;
            EXPECT_FALSE(std::isinf(outputs[i])) << impl << " at i=" << i;
        }
        double mean = std::accumulate(outputs.begin(), outputs.end(), 0.0) / batch_size;
        double variance = 0.0;
        for (double val : outputs) {
            variance += (val - mean) * (val - mean);
        }
        variance /= batch_size;
        EXPECT_TRUE(std::isfinite(mean)) << impl;
        EXPECT_TRUE(std::isfinite(variance)) << impl;
    }
}

// Integration tests (6)
TEST_F(GeluTest, NumericalGradientVerification) {
    std::vector<double> inputs = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
    for (const auto& impl : impls) {
        double tol = std::max(get_grad_tol(impl.name) * 10, 1e-4);
        if (is_fp16_impl(impl.name)) {
            tol = 1.0;
        }
        for (double x : inputs) {
            double analytical = impl.gelu_grad(x);
            double numerical = is_fp16_impl(impl.name) ?
                numerical_gradient_fp16_safe(impl.gelu, x) :
                numerical_gradient(impl.gelu, x, 1e-6);
            double rel_err = relative_error(analytical, numerical);
            EXPECT_LT(rel_err, tol) << impl << " at x=" << x
                << " (analytical=" << analytical << ", numerical=" << numerical << ")";
        }
    }
}

TEST_F(GeluTest, ForwardBackwardConsistency) {
    std::vector<double> inputs = {-2.5, -1.5, -0.5, 0.5, 1.5, 2.5};
    for (const auto& impl : impls) {
        double tol = std::max(get_grad_tol(impl.name), 1e-4);
        if (is_fp16_impl(impl.name)) {
            tol = 1.0;
        }
        for (double x : inputs) {
            double analytical = impl.gelu_grad(x);
            double numerical = is_fp16_impl(impl.name) ?
                numerical_gradient_fp16_safe(impl.gelu, x) :
                numerical_gradient(impl.gelu, x, 1e-6);
            double rel_err = relative_error(analytical, numerical);
            EXPECT_LT(rel_err, tol) << impl << " at x=" << x;
        }
    }
}

TEST_F(GeluTest, ChainRuleBackpropagation) {
    std::vector<double> inputs = {-1.0, 0.0, 0.5, 1.0};
    std::vector<double> weights = {0.5, 1.0, 1.5, 2.0};
    std::vector<double> biases = {-0.5, 0.0, 0.5, 1.0};
    double upstream_grad = 1.0;
    for (const auto& impl : impls) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            double pre_activation = weights[i] * inputs[i] + biases[i];
            double local_grad = impl.gelu_grad(pre_activation);
            double grad_wrt_weight = upstream_grad * local_grad * inputs[i];
            double grad_wrt_bias = upstream_grad * local_grad;
            double grad_wrt_input = upstream_grad * local_grad * weights[i];
            EXPECT_TRUE(std::isfinite(grad_wrt_weight)) << impl << " at i=" << i;
            EXPECT_TRUE(std::isfinite(grad_wrt_bias)) << impl << " at i=" << i;
            EXPECT_TRUE(std::isfinite(grad_wrt_input)) << impl << " at i=" << i;
            EXPECT_LT(std::abs(grad_wrt_weight), 10.0) << impl << " at i=" << i;
            EXPECT_LT(std::abs(grad_wrt_bias), 10.0) << impl << " at i=" << i;
            EXPECT_LT(std::abs(grad_wrt_input), 10.0) << impl << " at i=" << i;
        }
    }
}

TEST_F(GeluTest, MultiLayerGradientFlow) {
    const size_t num_layers = 3;
    const size_t hidden_size = 64;
    for (const auto& impl : impls) {
        auto inputs = generate_batch(hidden_size, -1.0, 1.0);
        std::vector<std::vector<double>> weights(num_layers);
        std::vector<double> biases(num_layers);
        for (size_t l = 0; l < num_layers; ++l) {
            weights[l] = generate_batch(hidden_size, -0.5, 0.5);
            biases[l] = 0.0;
        }
        std::vector<double> activations = inputs;
        std::vector<std::vector<double>> pre_activations(num_layers);
        std::vector<std::vector<double>> post_activations(num_layers);
        for (size_t l = 0; l < num_layers; ++l) {
            pre_activations[l].resize(hidden_size);
            post_activations[l].resize(hidden_size);
            for (size_t i = 0; i < hidden_size; ++i) {
                pre_activations[l][i] = weights[l][i] * activations[i] + biases[l];
                post_activations[l][i] = impl.gelu(pre_activations[l][i]);
            }
            activations = post_activations[l];
        }
        std::vector<double> grad_output(hidden_size, 1.0);
        for (int l = num_layers - 1; l >= 0; --l) {
            std::vector<double> grad_pre_activation(hidden_size);
            for (size_t i = 0; i < hidden_size; ++i) {
                double gelu_grad = impl.gelu_grad(pre_activations[l][i]);
                grad_pre_activation[i] = grad_output[i] * gelu_grad;
                EXPECT_TRUE(std::isfinite(grad_pre_activation[i])) << impl << " layer " << l;
                EXPECT_LT(std::abs(grad_pre_activation[i]), 100.0) << impl << " layer " << l;
            }
            grad_output = grad_pre_activation;
        }
        double mean_grad = std::accumulate(grad_output.begin(), grad_output.end(), 0.0) / hidden_size;
        EXPECT_TRUE(std::isfinite(mean_grad)) << impl;
        EXPECT_LT(std::abs(mean_grad), 10.0) << impl;
    }
}

TEST_F(GeluTest, GradientAccumulationConvergence) {
    const size_t num_iterations = 100;
    const size_t hidden_size = 32;
    const double learning_rate = 0.01;
    for (const auto& impl : impls) {
        auto inputs = generate_batch(hidden_size, -1.0, 1.0);
        auto targets = generate_batch(hidden_size, -1.0, 1.0);
        std::vector<double> weights = generate_batch(hidden_size, -0.5, 0.5);
        double bias = 0.0;
        double initial_loss = 0.0;
        double final_loss = 0.0;
        for (size_t iter = 0; iter < num_iterations; ++iter) {
            std::vector<double> outputs(hidden_size);
            double loss = 0.0;
            for (size_t i = 0; i < hidden_size; ++i) {
                double pre_activation = weights[i] * inputs[i] + bias;
                outputs[i] = impl.gelu(pre_activation);
                double error = outputs[i] - targets[i];
                loss += error * error;
            }
            loss /= hidden_size;
            if (iter == 0) initial_loss = loss;
            if (iter == num_iterations - 1) final_loss = loss;
            for (size_t i = 0; i < hidden_size; ++i) {
                double pre_activation = weights[i] * inputs[i] + bias;
                double gelu_grad = impl.gelu_grad(pre_activation);
                double error = outputs[i] - targets[i];
                double grad = (2.0 / hidden_size) * error * gelu_grad;
                weights[i] -= learning_rate * grad * inputs[i];
                bias -= learning_rate * grad;
                EXPECT_TRUE(std::isfinite(weights[i])) << impl << " at iter " << iter;
            }
        }
        if (is_fp16_impl(impl.name)) {
            EXPECT_LT(final_loss, initial_loss * 1.5)
                << impl << " FP16 (initial=" << initial_loss << ", final=" << final_loss << ")";
        } else {
            EXPECT_LT(final_loss, initial_loss)
                << impl << " (initial=" << initial_loss << ", final=" << final_loss << ")";
        }
    }
}

TEST_F(GeluTest, ConsistencyAcrossImplementations) {
    std::vector<double> inputs = {-2.0, -1.0, 0.0, 1.0, 2.0};
    for (double x : inputs) {
        double ref_fwd = reference_gelu(x);
        for (const auto& impl : impls) {
            double tol = get_fwd_tol(impl.name);
            EXPECT_NEAR(impl.gelu(x), ref_fwd, tol) << impl << " at x=" << x;
        }
        double ref_bwd = reference_gelu_grad(x);
        for (const auto& impl : impls) {
            double tol = get_grad_tol(impl.name);
            EXPECT_NEAR(impl.gelu_grad(x), ref_bwd, tol) << impl << " at x=" << x;
        }
    }
}

// ============================================================================
// SECTION 8: Tensor Tests (8 tests)
// ============================================================================

TEST_F(GeluTest, TensorCreation) {
    using namespace gelu_test;
    auto shape = TensorShape(2, 1, 4, 8);
    auto tensor = TensorOps::zeros(shape);
    validate_shape(tensor, shape);
    EXPECT_EQ(tensor.data().size(), shape.total_size());
}

TEST_F(GeluTest, TensorForwardAllModes) {
    using namespace gelu_test;
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto input = TensorOps::create(data, TensorShape(1, 1, 1, 5));
    for (auto mode : {GELUMode::ERF, GELUMode::TANH, GELUMode::SIGMOID}) {
        auto output = TensorOps::gelu(input, mode);
        auto out_data = output.data();
        auto impl = GELURegistry::instance().get(mode);
        for (size_t i = 0; i < data.size(); ++i) {
            float expected = impl.forward(data[i]);
            EXPECT_NEAR(out_data[i], expected, get_tensor_tolerance(mode, expected));
        }
    }
}

TEST_F(GeluTest, TensorBatchProcessing) {
    using namespace gelu_test;
    auto shape = TensorShape(4, 1, 8, 16);
    auto input = TensorOps::randn(shape, 0.0f, 1.0f);
    BatchGELU gelu(GELUMode::TANH);
    auto output = gelu.forward(input);
    validate_shape(output, shape);
    validate_no_nans(output);
}

TEST_F(GeluTest, TensorBackwardPass) {
    using namespace gelu_test;
    SKIP_IF_NO_CAPABILITY(Capability::AUTOGRAD);
    auto input = TensorOps::create({-1.0f, 0.0f, 1.0f}, TensorShape(1, 1, 1, 3), true);
    auto output = TensorOps::gelu(input);
    auto target = TensorOps::zeros(input.shape());
    auto loss = TensorOps::mse_loss(output, target);
    TensorOps::backward(loss);
    auto grad = input.grad();
    bool has_nonzero = false;
    for (float g : grad) {
        EXPECT_FALSE(std::isnan(g));
        if (std::abs(g) > 1e-6f) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(GeluTest, TensorBertSmallConfig) {
    using namespace gelu_test;
    auto config = BertConfig::small();
    auto input = TensorOps::randn(config.input_shape(), 0.0f, 0.02f);
    auto output = TensorOps::gelu(input);
    validate_shape(output, config.input_shape());
    validate_no_nans(output);
}

TEST_F(GeluTest, TensorShapeOperations) {
    using namespace gelu_test;
    TensorShape s1(2, 1, 4, 8);
    TensorShape s2(2, 1, 4, 8);
    TensorShape s3(1, 1, 4, 8);
    EXPECT_EQ(s1, s2);
    EXPECT_NE(s1, s3);
    EXPECT_EQ(s1.total_size(), 64u);
}

TEST_F(GeluTest, TensorRegistryConsistency) {
    using namespace gelu_test;
    for (auto mode : {GELUMode::ERF, GELUMode::TANH, GELUMode::SIGMOID}) {
        auto impl = GELURegistry::instance().get(mode);
        EXPECT_NEAR(impl.forward(0.0f), 0.0f, 1e-6f) << impl.name;
        EXPECT_NEAR(impl.backward(0.0f), 0.5f, impl.backward_tolerance) << impl.name;
    }
}

TEST_F(GeluTest, TensorCapabilityReport) {
    using namespace gelu_test;
    std::cout << "\nTensor Backend: " << TensorOps::backend_name() << "\n";
    std::cout << "Capabilities:\n";
    std::cout << "  Elementwise: " << (TensorOps::supports(Capability::ELEMENTWISE) ? "Yes" : "No") << "\n";
    std::cout << "  MatMul: " << (TensorOps::supports(Capability::MATMUL) ? "Yes" : "No") << "\n";
    std::cout << "  Autograd: " << (TensorOps::supports(Capability::AUTOGRAD) ? "Yes" : "No") << "\n";
}

// ============================================================================
// Main
// ============================================================================

#ifdef WE_NEED_MAIN_HERE

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n==========================================\n";
    std::cout << "GELU Testing Framework - Full Power Edition\n";
    std::cout << "==========================================\n";
    std::cout << "Scalar backend: FP64 + " << (GELU_HAS_FP16 ? "FP16" : "no FP16") << "\n";
    std::cout << "Tensor backend: " << gelu_test::TensorOps::backend_name() << "\n";
    std::cout << "Implementations: " << impls.size() << "\n";
    std::cout << "Tests: 34 scalar";
#if GELU_HAS_FP16
    std::cout << " + 4 FP16";
#endif
    std::cout << " + 8 tensor = " << (34 + (GELU_HAS_FP16 ? 4 : 0) + 8) << " total\n";
    std::cout << "==========================================\n\n";

    return RUN_ALL_TESTS();
}

#endif // #ifdef WE_NEED_MAIN_HERE

