// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#include "core/random.hpp"
#include "tt-metalium/bfloat16.hpp"

namespace {

// Statistical analysis structure
struct Statistics {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;

    void compute(std::vector<double>& data) {
        if (data.empty())
            return;

        // Mean
        mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

        // Standard deviation
        double variance = 0.0;
        for (double x : data) {
            variance += (x - mean) * (x - mean);
        }
        stddev = std::sqrt(variance / data.size());

        // Min/Max
        auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
        min = *min_it;
        max = *max_it;

        // Median
        std::vector<double> sorted = data;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.size() % 2 == 0) {
            median = (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0;
        } else {
            median = sorted[sorted.size() / 2];
        }
    }

    void print(const std::string& label) const {
        std::cout << label << ":\n";
        std::cout << "  Mean:     " << std::fixed << std::setprecision(6) << mean << "\n";
        std::cout << "  Std Dev:  " << std::fixed << std::setprecision(6) << stddev << "\n";
        std::cout << "  Min:      " << std::fixed << std::setprecision(6) << min << "\n";
        std::cout << "  Max:      " << std::fixed << std::setprecision(6) << max << "\n";
        std::cout << "  Median:   " << std::fixed << std::setprecision(6) << median << "\n";
    }
};

// Kolmogorov-Smirnov test (simplified version)
double ks_statistic(std::vector<double>& data1, std::vector<double>& data2) {
    std::sort(data1.begin(), data1.end());
    std::sort(data2.begin(), data2.end());

    double max_diff = 0.0;
    size_t i = 0, j = 0;

    while (i < data1.size() && j < data2.size()) {
        double cdf1 = static_cast<double>(i) / data1.size();
        double cdf2 = static_cast<double>(j) / data2.size();
        max_diff = std::max(max_diff, std::abs(cdf1 - cdf2));

        if (data1[i] <= data2[j]) {
            i++;
        } else {
            j++;
        }
    }

    // Account for remaining elements
    while (i < data1.size()) {
        double cdf1 = static_cast<double>(i) / data1.size();
        max_diff = std::max(max_diff, std::abs(cdf1 - 1.0));
        i++;
    }

    while (j < data2.size()) {
        double cdf2 = static_cast<double>(j) / data2.size();
        max_diff = std::max(max_diff, std::abs(1.0 - cdf2));
        j++;
    }

    return max_diff;
}

// Validate that a distribution matches expected properties
struct DistributionValidator {
    double expected_mean;
    double expected_stddev;
    double tolerance_mean_pct;    // tolerance for mean in percentage
    double tolerance_stddev_pct;  // tolerance for stddev in percentage
    double range_min;
    double range_max;

    bool validate(const Statistics& stats, std::string& error_msg) const {
        error_msg.clear();

        // Check mean - use absolute error for values near zero, percentage for others
        double mean_diff = std::abs(stats.mean - expected_mean);
        double mean_error_pct;
        if (std::abs(expected_mean) > 0.1) {
            mean_error_pct = mean_diff / std::abs(expected_mean) * 100.0;
        } else {
            // For near-zero means, use absolute error tolerance converted to percentage
            mean_error_pct = mean_diff * 100.0;
        }
        if (mean_error_pct > tolerance_mean_pct) {
            error_msg += "  ✗ Mean out of tolerance: " + std::to_string(mean_error_pct) + "% error\n";
            return false;
        }

        // Check standard deviation - use absolute error for values near zero, percentage for others
        double stddev_diff = std::abs(stats.stddev - expected_stddev);
        double stddev_error_pct;
        if (std::abs(expected_stddev) > 0.1) {
            stddev_error_pct = stddev_diff / std::abs(expected_stddev) * 100.0;
        } else {
            // For near-zero stddevs, use absolute error tolerance converted to percentage
            stddev_error_pct = stddev_diff * 100.0;
        }
        if (stddev_error_pct > tolerance_stddev_pct) {
            error_msg += "  ✗ Std Dev out of tolerance: " + std::to_string(stddev_error_pct) + "% error\n";
            return false;
        }

        // Check range
        if (stats.min < range_min || stats.max > range_max) {
            error_msg += "  ✗ Values out of expected range: [" + std::to_string(stats.min) + ", " +
                         std::to_string(stats.max) + "]\n";
            return false;
        }

        return true;
    }

    void print_status(const Statistics& stats, bool is_valid, const std::string& error_msg) const {
        if (is_valid) {
            std::cout << "  ✓ Distribution is VALID\n";

            // Calculate mean error using same logic as validate()
            double mean_diff = std::abs(stats.mean - expected_mean);
            double mean_error_pct;
            if (std::abs(expected_mean) > 0.1) {
                mean_error_pct = mean_diff / std::abs(expected_mean) * 100.0;
            } else {
                mean_error_pct = mean_diff * 100.0;
            }
            std::cout << "    Mean error: " << std::fixed << std::setprecision(4) << mean_error_pct << "%\n";

            // Calculate stddev error using same logic as validate()
            double stddev_diff = std::abs(stats.stddev - expected_stddev);
            double stddev_error_pct;
            if (std::abs(expected_stddev) > 0.1) {
                stddev_error_pct = stddev_diff / std::abs(expected_stddev) * 100.0;
            } else {
                stddev_error_pct = stddev_diff * 100.0;
            }
            std::cout << "    Stddev error: " << std::fixed << std::setprecision(4) << stddev_error_pct << "%\n";
        } else {
            std::cout << "  ✗ Distribution is INVALID:\n" << error_msg;
        }
    }
};

}  // namespace

int main() {
    const uint32_t seed = 42;
    const size_t num_samples = 1000000;

    std::cout << "==========================================================================\n";
    std::cout << "Random Number Generation Distribution Comparison\n";
    std::cout << "==========================================================================\n";
    std::cout << "Seed: " << seed << "\n";
    std::cout << "Samples: " << num_samples << "\n\n";

    // Test 1: Uniform Distribution (float)
    {
        std::cout << "Test 1: Uniform Distribution [0, 1) - float\n";
        std::cout << "----------\n";

        std::vector<float> legacy_data(num_samples);
        std::vector<float> sse_data(num_samples);

        // Generate using legacy implementation
        ttml::core::legacy::sequential_generate(
            std::span<float>{legacy_data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

        // Generate using SSE implementation
        ttml::core::sse::sequential_generate(
            std::span<float>{sse_data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

        // Convert to double for analysis
        std::vector<double> legacy_double(legacy_data.begin(), legacy_data.end());
        std::vector<double> sse_double(sse_data.begin(), sse_data.end());

        Statistics legacy_stats, sse_stats;
        legacy_stats.compute(legacy_double);
        sse_stats.compute(sse_double);

        legacy_stats.print("Legacy Implementation");
        std::cout << "\n";
        sse_stats.print("SSE Implementation");

        // Validate distributions
        // For Uniform[0,1): mean = 0.5, stddev = 1/sqrt(12) ≈ 0.2887
        DistributionValidator validator{0.5, 0.2887, 1.0, 5.0, 0.0, 1.0};
        std::string legacy_error, sse_error;
        bool legacy_valid = validator.validate(legacy_stats, legacy_error);
        bool sse_valid = validator.validate(sse_stats, sse_error);

        std::cout << "\nValidation:\n";
        std::cout << "Legacy: ";
        validator.print_status(legacy_stats, legacy_valid, legacy_error);
        std::cout << "SSE:    ";
        validator.print_status(sse_stats, sse_valid, sse_error);

        // KS test
        double ks = ks_statistic(legacy_double, sse_double);
        std::cout << "\nKolmogorov-Smirnov Statistic: " << std::fixed << std::setprecision(6) << ks << "\n";
        std::cout << "(Lower values indicate more similar distributions)\n\n";
    }

    // Test 2: Normal Distribution (float)
    {
        std::cout << "Test 2: Normal Distribution (mean=0, stddev=1) - float\n";
        std::cout << "----------\n";

        std::vector<float> legacy_data(num_samples);
        std::vector<float> sse_data(num_samples);

        // Generate using legacy implementation
        ttml::core::legacy::sequential_generate(
            std::span<float>{legacy_data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

        // Generate using SSE implementation
        ttml::core::sse::sequential_generate(
            std::span<float>{sse_data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

        // Convert to double for analysis
        std::vector<double> legacy_double(legacy_data.begin(), legacy_data.end());
        std::vector<double> sse_double(sse_data.begin(), sse_data.end());

        Statistics legacy_stats, sse_stats;
        legacy_stats.compute(legacy_double);
        sse_stats.compute(sse_double);

        legacy_stats.print("Legacy Implementation");
        std::cout << "\n";
        sse_stats.print("SSE Implementation");

        // Validate distributions
        // For Normal(0,1): mean = 0, stddev = 1
        DistributionValidator validator{0.0, 1.0, 1.0, 5.0, -10.0, 10.0};
        std::string legacy_error, sse_error;
        bool legacy_valid = validator.validate(legacy_stats, legacy_error);
        bool sse_valid = validator.validate(sse_stats, sse_error);

        std::cout << "\nValidation:\n";
        std::cout << "Legacy: ";
        validator.print_status(legacy_stats, legacy_valid, legacy_error);
        std::cout << "SSE:    ";
        validator.print_status(sse_stats, sse_valid, sse_error);

        // KS test
        double ks = ks_statistic(legacy_double, sse_double);
        std::cout << "\nKolmogorov-Smirnov Statistic: " << std::fixed << std::setprecision(6) << ks << "\n";
        std::cout << "(Lower values indicate more similar distributions)\n\n";
    }

    // Test 3: Uniform Distribution (bfloat16)
    {
        std::cout << "Test 3: Uniform Distribution [0, 1) - bfloat16\n";
        std::cout << "----------\n";

        std::vector<bfloat16> legacy_data(num_samples);
        std::vector<bfloat16> sse_data(num_samples);

        // Generate using legacy implementation
        ttml::core::legacy::sequential_generate(
            std::span<bfloat16>{legacy_data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

        // Generate using SSE implementation
        ttml::core::sse::sequential_generate(
            std::span<bfloat16>{sse_data}, []() { return std::uniform_real_distribution<float>(0.0f, 1.0f); }, seed);

        // Convert to double for analysis
        std::vector<double> legacy_double;
        std::vector<double> sse_double;
        for (auto val : legacy_data) {
            legacy_double.push_back(static_cast<double>(static_cast<float>(val)));
        }
        for (auto val : sse_data) {
            sse_double.push_back(static_cast<double>(static_cast<float>(val)));
        }

        Statistics legacy_stats, sse_stats;
        legacy_stats.compute(legacy_double);
        sse_stats.compute(sse_double);

        legacy_stats.print("Legacy Implementation");
        std::cout << "\n";
        sse_stats.print("SSE Implementation");

        // Validate distributions
        // For Uniform[0,1): mean = 0.5, stddev = 1/sqrt(12) ≈ 0.2887
        // Higher tolerance for bfloat16 due to lower precision
        DistributionValidator validator{0.5, 0.2887, 2.0, 10.0, 0.0, 1.0};
        std::string legacy_error, sse_error;
        bool legacy_valid = validator.validate(legacy_stats, legacy_error);
        bool sse_valid = validator.validate(sse_stats, sse_error);

        std::cout << "\nValidation:\n";
        std::cout << "Legacy: ";
        validator.print_status(legacy_stats, legacy_valid, legacy_error);
        std::cout << "SSE:    ";
        validator.print_status(sse_stats, sse_valid, sse_error);

        // KS test
        double ks = ks_statistic(legacy_double, sse_double);
        std::cout << "\nKolmogorov-Smirnov Statistic: " << std::fixed << std::setprecision(6) << ks << "\n";
        std::cout << "(Lower values indicate more similar distributions)\n\n";
    }

    // Test 4: Normal Distribution (bfloat16)
    {
        std::cout << "Test 4: Normal Distribution (mean=0, stddev=1) - bfloat16\n";
        std::cout << "----------\n";

        std::vector<bfloat16> legacy_data(num_samples);
        std::vector<bfloat16> sse_data(num_samples);

        // Generate using legacy implementation
        ttml::core::legacy::sequential_generate(
            std::span<bfloat16>{legacy_data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

        // Generate using SSE implementation
        ttml::core::sse::sequential_generate(
            std::span<bfloat16>{sse_data}, []() { return std::normal_distribution<float>(0.0f, 1.0f); }, seed);

        // Convert to double for analysis
        std::vector<double> legacy_double;
        std::vector<double> sse_double;
        for (auto val : legacy_data) {
            legacy_double.push_back(static_cast<double>(static_cast<float>(val)));
        }
        for (auto val : sse_data) {
            sse_double.push_back(static_cast<double>(static_cast<float>(val)));
        }

        Statistics legacy_stats, sse_stats;
        legacy_stats.compute(legacy_double);
        sse_stats.compute(sse_double);

        legacy_stats.print("Legacy Implementation");
        std::cout << "\n";
        sse_stats.print("SSE Implementation");

        // Validate distributions
        // For Normal(0,1): mean = 0, stddev = 1
        // Higher tolerance for bfloat16 due to lower precision
        DistributionValidator validator{0.0, 1.0, 2.0, 10.0, -10.0, 10.0};
        std::string legacy_error, sse_error;
        bool legacy_valid = validator.validate(legacy_stats, legacy_error);
        bool sse_valid = validator.validate(sse_stats, sse_error);

        std::cout << "\nValidation:\n";
        std::cout << "Legacy: ";
        validator.print_status(legacy_stats, legacy_valid, legacy_error);
        std::cout << "SSE:    ";
        validator.print_status(sse_stats, sse_valid, sse_error);

        // KS test
        double ks = ks_statistic(legacy_double, sse_double);
        std::cout << "\nKolmogorov-Smirnov Statistic: " << std::fixed << std::setprecision(6) << ks << "\n";
        std::cout << "(Lower values indicate more similar distributions)\n\n";
    }

    std::cout << "==========================================================================\n";
    std::cout << "Comparison Complete\n";
    std::cout << "==========================================================================\n";

    return 0;
}
