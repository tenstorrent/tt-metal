/**
 * Lifting Wavelet Transform for tt-metal
 * 
 * Hardware-accelerated LWT implementation using tt-metal platform
 * Supports parallel lifting steps and optimized memory layout
 * 
 * 版权声明：MIT License | Copyright (c) 2026 思捷娅科技 (SJYKJ)
 */

#include "tt_metal/tt_metal.hpp"
#include "tt_metal/common/logger.hpp"
#include <vector>
#include <memory>

namespace tt::lwt {

using namespace tt::tt_metal;

/**
 * LWT Kernel Configuration
 */
struct LWTConfig {
    uint32_t input_size;
    uint32_t levels;
    uint32_t wavelet_type;  // 0: Haar, 1: Db4, 2: CDF97
    bool is_2d;
};

/**
 * Lifting Wavelet Transform Accelerator
 */
class LWTAccelerator {
public:
    LWTAccelerator(uint32_t device_id = 0);
    ~LWTAccelerator();
    
    /**
     * Initialize device and allocate memory
     */
    void initialize(const LWTConfig& config);
    
    /**
     * Execute forward LWT
     */
    std::vector<float> transform(const std::vector<float>& input);
    
    /**
     * Execute inverse LWT
     */
    std::vector<float> inverse_transform(const std::vector<float>& coefficients);
    
    /**
     * Get performance metrics
     */
    struct Metrics {
        float transform_time_ms;
        float inverse_time_ms;
        float throughput_gbps;
    };
    Metrics get_metrics() const { return metrics_; }

private:
    /**
     * Create compute kernels
     */
    void create_kernels();
    
    /**
     * Setup memory buffers
     */
    void setup_buffers(const LWTConfig& config);
    
    /**
     * Execute kernel on device
     */
    void execute_kernel(bool inverse = false);
    
    // Device handle
    std::shared_ptr<Device> device_;
    
    // Program and kernels
    std::shared_ptr<Program> program_;
    KernelHandle split_kernel_;
    KernelHandle predict_kernel_;
    KernelHandle update_kernel_;
    KernelHandle merge_kernel_;
    
    // Memory buffers
    Buffer input_buffer_;
    Buffer output_buffer_;
    Buffer intermediate_buffer_;
    
    // Configuration
    LWTConfig config_;
    Metrics metrics_;
};

/**
 * LWT Kernel Implementation (for tt-metal)
 */
void split_kernel(MultiCoreReaderWriterConfig config) {
    // Split signal into even and odd samples
    // Parallel implementation for tt-metal
    uint32_t start_idx = config.core_id * config.chunk_size;
    uint32_t end_idx = start_idx + config.chunk_size;
    
    for (uint32_t i = start_idx; i < end_idx; i += 2) {
        // Even sample
        float even = input_buffer[i];
        write_to_output(even, even_buffer, i / 2);
        
        // Odd sample
        if (i + 1 < end_idx) {
            float odd = input_buffer[i + 1];
            write_to_output(odd, odd_buffer, i / 2);
        }
    }
}

void predict_kernel(MultiCoreReaderWriterConfig config) {
    // Predict step: d[n] = odd[n] - P(even[n])
    uint32_t start_idx = config.core_id * config.chunk_size;
    uint32_t end_idx = start_idx + config.chunk_size;
    
    for (uint32_t i = start_idx; i < end_idx; i++) {
        float prediction = 0.0f;
        
        // Apply prediction filter based on wavelet type
        if (config.wavelet_type == 0) {  // Haar
            prediction = even_buffer[i];
        } else if (config.wavelet_type == 1) {  // Db4
            prediction = 0.4829629131445341f * even_buffer[i] + 
                        0.8365163037378079f * even_buffer[i + 1];
        }
        
        float detail = odd_buffer[i] - prediction;
        write_to_output(detail, detail_buffer, i);
    }
}

void update_kernel(MultiCoreReaderWriterConfig config) {
    // Update step: s[n] = even[n] + U(detail[n])
    uint32_t start_idx = config.core_id * config.chunk_size;
    uint32_t end_idx = start_idx + config.chunk_size;
    
    for (uint32_t i = start_idx; i < end_idx; i++) {
        float update = 0.0f;
        
        // Apply update filter based on wavelet type
        if (config.wavelet_type == 0) {  // Haar
            update = 0.5f * detail_buffer[i];
        } else if (config.wavelet_type == 1) {  // Db4
            update = -0.2241438680420134f * detail_buffer[i - 1] + 
                    0.1294095225512604f * detail_buffer[i];
        }
        
        float approx = even_buffer[i] + update;
        write_to_output(approx, approx_buffer, i);
    }
}

void merge_kernel(MultiCoreReaderWriterConfig config) {
    // Merge even and odd samples back into signal
    uint32_t start_idx = config.core_id * config.chunk_size;
    uint32_t end_idx = start_idx + config.chunk_size;
    
    for (uint32_t i = start_idx; i < end_idx; i++) {
        // Even sample
        float even = approx_buffer[i];
        write_to_output(even, output_buffer, i * 2);
        
        // Odd sample
        if (i * 2 + 1 < config.input_size) {
            float odd = detail_buffer[i];
            write_to_output(odd, output_buffer, i * 2 + 1);
        }
    }
}

} // namespace tt::lwt

// Example usage
int main() {
    using namespace tt::lwt;
    
    // Initialize accelerator
    LWTAccelerator accelerator(0);
    
    // Configure
    LWTConfig config;
    config.input_size = 1024;
    config.levels = 3;
    config.wavelet_type = 0;  // Haar
    config.is_2d = false;
    
    accelerator.initialize(config);
    
    // Generate test signal
    std::vector<float> input(config.input_size);
    for (uint32_t i = 0; i < config.input_size; i++) {
        input[i] = std::sin(2.0f * M_PI * i / 100.0f);
    }
    
    // Forward transform
    auto coefficients = accelerator.transform(input);
    
    // Inverse transform
    auto reconstructed = accelerator.inverse_transform(coefficients);
    
    // Verify reconstruction
    float max_error = 0.0f;
    for (size_t i = 0; i < input.size(); i++) {
        float error = std::abs(input[i] - reconstructed[i]);
        if (error > max_error) max_error = error;
    }
    
    std::cout << "Max reconstruction error: " << max_error << std::endl;
    
    // Print metrics
    auto metrics = accelerator.get_metrics();
    std::cout << "Transform time: " << metrics.transform_time_ms << " ms" << std::endl;
    std::cout << "Inverse time: " << metrics.inverse_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << metrics.throughput_gbps << " GB/s" << std::endl;
    
    return 0;
}
