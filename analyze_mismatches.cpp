#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <sstream>
#include <cinttypes>
#include <bit>

// Structure to hold bfloat16 analysis
struct BFloat16Info {
    uint16_t raw_value;
    bool sign;
    uint8_t exponent;
    uint8_t mantissa;
    bool is_nan;
    bool is_inf;
    bool is_subnormal;
    bool is_zero;

    BFloat16Info(uint16_t value) : raw_value(value) {
        // Extract fields: sign(1) | exponent(8) | mantissa(7)
        sign = (value & 0x8000) != 0;
        exponent = (value >> 7) & 0xFF;
        mantissa = value & 0x7F;

        // Check special cases
        is_zero = (exponent == 0 && mantissa == 0);
        is_subnormal = (exponent == 0 && mantissa != 0);
        is_inf = (exponent == 0xFF && mantissa == 0);
        is_nan = (exponent == 0xFF && mantissa != 0);
    }

    std::string get_type() const {
        if (is_nan) {
            return "NaN";
        }
        if (is_inf) {
            return sign ? "-Inf" : "+Inf";
        }
        if (is_zero) {
            return sign ? "-Zero" : "+Zero";
        }
        if (is_subnormal) {
            return "Subnormal";
        }
        return "Normal";
    }
};

// Calculate ULP (Units in the Last Place) difference between two bfloat16 values
// represented as uint16_t
uint64_t calculate_ulp_difference(uint16_t value1, uint16_t value2) {
    // For bfloat16, the bit representation is: sign(1) | exponent(8) | mantissa(7)

    // Handle special cases
    // If both values are the same, ULP difference is 0
    if (value1 == value2) {
        return 0;
    }

    // Extract sign bits
    bool sign1 = (value1 & 0x8000) != 0;
    bool sign2 = (value2 & 0x8000) != 0;

    // If signs differ, we need to handle this specially
    // The ULP distance crosses zero
    if (sign1 != sign2) {
        // Distance from value1 to zero + distance from zero to value2
        uint64_t dist1 = value1 & 0x7FFF;  // Distance to zero (remove sign bit)
        uint64_t dist2 = value2 & 0x7FFF;  // Distance to zero (remove sign bit)
        return dist1 + dist2;
    }

    // Same sign: simple subtraction
    if (value1 > value2) {
        return static_cast<uint64_t>(value1 - value2);
    } else {
        return static_cast<uint64_t>(value2 - value1);
    }
}

void print_frequency_table(
    const std::vector<std::pair<uint16_t, uint64_t>>& frequency_vec,
    uint64_t total_count,
    const std::string& title,
    int num_to_display) {
    std::cout << "\n" << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::setw(6) << "Rank" << std::setw(12) << "Value (hex)" << std::setw(12) << "Value (dec)"
              << std::setw(15) << "Frequency" << std::setw(15) << "Percentage" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int i = 0; i < num_to_display; ++i) {
        double percentage = (static_cast<double>(frequency_vec[i].second) / total_count) * 100.0;

        std::cout << std::setw(6) << (i + 1) << std::setw(12) << "0x" << std::hex << std::setfill('0') << std::setw(4)
                  << frequency_vec[i].first << std::dec << std::setfill(' ') << std::setw(12) << frequency_vec[i].first
                  << std::setw(15) << frequency_vec[i].second << std::setw(14) << std::fixed << std::setprecision(2)
                  << percentage << "%" << std::endl;
    }

    std::cout << std::string(60, '=') << std::endl;

    // Calculate what percentage of total the top entries represent
    uint64_t top_count = 0;
    for (int i = 0; i < num_to_display; ++i) {
        top_count += frequency_vec[i].second;
    }

    double top_percentage = (static_cast<double>(top_count) / total_count) * 100.0;
    std::cout << "\nTop " << num_to_display << " values account for " << top_count << " entries (" << std::fixed
              << std::setprecision(2) << top_percentage << "% of total)" << std::endl;
}

// Structure to hold a single mismatch record
struct MismatchRecord {
    uint16_t input_a;
    uint16_t input_b;
    uint16_t result;
    uint16_t expected;
};

// Function to generate CSV file for a specific left operand value
void generate_csv_for_operand(
    uint16_t operand_a_value, const std::vector<MismatchRecord>& records, const std::string& output_dir) {
    // Create filename
    std::ostringstream filename;
    filename << output_dir << "/operand_a_0x" << std::hex << std::setfill('0') << std::setw(4) << operand_a_value
             << ".csv";

    std::ofstream csv_file(filename.str());
    if (!csv_file.is_open()) {
        std::cerr << "Warning: Could not create CSV file: " << filename.str() << std::endl;
        return;
    }

    // Write header
    csv_file << "Left Operand (hex),Right Operand (hex),Result (hex),Expected (hex),"
             << "ULP Error,Signs Same,Larger Exponent,Smaller Exponent,Exponent Diff,"
             << "Left Type,Right Type,Left Exp,Right Exp,Left Mantissa,Right Mantissa\n";

    // Process each record
    for (const auto& record : records) {
        BFloat16Info left(record.input_a);
        BFloat16Info right(record.input_b);
        BFloat16Info result(record.result);
        BFloat16Info expected(record.expected);

        uint64_t ulp_error = calculate_ulp_difference(record.result, record.expected);

        // Check if signs are the same
        bool signs_same = (left.sign == right.sign);

        // Get larger and smaller exponents
        uint8_t larger_exp = std::max(left.exponent, right.exponent);
        uint8_t smaller_exp = std::min(left.exponent, right.exponent);
        int exp_diff = static_cast<int>(larger_exp) - static_cast<int>(smaller_exp);

        // Write data row
        csv_file << "0x" << std::hex << std::setfill('0') << std::setw(4) << record.input_a << ","
                 << "0x" << std::hex << std::setfill('0') << std::setw(4) << record.input_b << ","
                 << "0x" << std::hex << std::setfill('0') << std::setw(4) << record.result << ","
                 << "0x" << std::hex << std::setfill('0') << std::setw(4) << record.expected << "," << std::dec
                 << ulp_error << "," << (signs_same ? "Yes" : "No") << "," << static_cast<int>(larger_exp) << ","
                 << static_cast<int>(smaller_exp) << "," << exp_diff << "," << left.get_type() << ","
                 << right.get_type() << "," << static_cast<int>(left.exponent) << ","
                 << static_cast<int>(right.exponent) << "," << static_cast<int>(left.mantissa) << ","
                 << static_cast<int>(right.mantissa) << "\n";
    }

    csv_file.close();
}

// Function to generate CSV file with filtered mismatches
void generate_filtered_csv(
    const std::vector<MismatchRecord>& all_records, const std::string& output_dir, int max_records = 100) {
    // Filter records based on criteria:
    // 1) Same input sign
    // 2) Both normal numbers (not subnormal, not inf, not nan, not zero)
    // 3) Exponent difference = 9

    std::vector<MismatchRecord> filtered_records;

    for (const auto& record : all_records) {
        BFloat16Info left(record.input_a);
        BFloat16Info right(record.input_b);

        // Check if same sign
        bool same_sign = (left.sign == right.sign);

        // Check if both are normal numbers
        bool both_normal = (left.get_type() == "Normal") && (right.get_type() == "Normal");

        // Calculate exponent difference
        int exp_diff = std::abs(static_cast<int>(left.exponent) - static_cast<int>(right.exponent));

        if (same_sign && both_normal && exp_diff == 9) {
            filtered_records.push_back(record);
            if (filtered_records.size() >= static_cast<size_t>(max_records)) {
                break;
            }
        }
    }

    // Create filename
    std::string filename = output_dir + "/filtered_exp9_same_sign_normal.csv";

    std::ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        std::cerr << "Warning: Could not create filtered CSV file: " << filename << std::endl;
        return;
    }

    // Write header
    csv_file << "Left Operand (hex),Right Operand (hex),Result (hex),Expected (hex),"
             << "ULP Error,Signs Same,Larger Exponent,Smaller Exponent,Exponent Diff,"
             << "Left Type,Right Type,Left Exp,Right Exp,Left Mantissa,Right Mantissa\n";

    // Process each filtered record
    for (const auto& record : filtered_records) {
        BFloat16Info left(record.input_a);
        BFloat16Info right(record.input_b);
        BFloat16Info result(record.result);
        BFloat16Info expected(record.expected);

        uint64_t ulp_error = calculate_ulp_difference(record.result, record.expected);

        // Check if signs are the same
        bool signs_same = (left.sign == right.sign);

        // Get larger and smaller exponents
        uint8_t larger_exp = std::max(left.exponent, right.exponent);
        uint8_t smaller_exp = std::min(left.exponent, right.exponent);
        int exp_diff = static_cast<int>(larger_exp) - static_cast<int>(smaller_exp);

        // Write data row
        csv_file << "0x" << std::hex << std::setfill('0') << std::setw(4) << record.input_a << ","
                 << "0x" << std::hex << std::setfill('0') << std::setw(4) << record.input_b << ","
                 << "0x" << std::hex << std::setfill('0') << std::setw(4) << record.result << ","
                 << "0x" << std::hex << std::setfill('0') << std::setw(4) << record.expected << "," << std::dec
                 << ulp_error << "," << (signs_same ? "Yes" : "No") << "," << static_cast<int>(larger_exp) << ","
                 << static_cast<int>(smaller_exp) << "," << exp_diff << "," << left.get_type() << ","
                 << right.get_type() << "," << static_cast<int>(left.exponent) << ","
                 << static_cast<int>(right.exponent) << "," << static_cast<int>(left.mantissa) << ","
                 << static_cast<int>(right.mantissa) << "\n";
    }

    csv_file.close();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "FILTERED MISMATCH CSV GENERATION" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Filter criteria:" << std::endl;
    std::cout << "  - Same input sign: Yes" << std::endl;
    std::cout << "  - Both operands normal (not subnormal/inf/nan/zero): Yes" << std::endl;
    std::cout << "  - Exponent difference: 9" << std::endl;
    std::cout << "\nRecords found matching criteria: " << filtered_records.size() << std::endl;
    std::cout << "Records written to CSV: " << std::min(static_cast<int>(filtered_records.size()), max_records)
              << std::endl;
    std::cout << "Output file: " << filename << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

// Generic histogram printer for categorical data
void print_categorical_histogram(
    const std::map<std::string, uint64_t>& histogram, uint64_t total_count, const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // For type histograms, ensure all special value categories are included (even with 0 count)
    // Note: We don't add "Normal" as it would clutter output - it will always have some frequency
    std::map<std::string, uint64_t> complete_histogram = histogram;
    if (title.find("TYPE HISTOGRAM") != std::string::npos) {
        // Define special value categories (excluding Normal which is common)
        std::vector<std::string> special_types = {"Subnormal", "+Inf", "-Inf", "NaN", "+Zero", "-Zero"};
        for (const auto& type : special_types) {
            if (complete_histogram.find(type) == complete_histogram.end()) {
                complete_histogram[type] = 0;
            }
        }
    }

    // Convert to vector and sort by count (descending)
    std::vector<std::pair<std::string, uint64_t>> hist_vec(complete_histogram.begin(), complete_histogram.end());
    std::sort(hist_vec.begin(), hist_vec.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    // Verify that frequencies sum to total count
    uint64_t frequency_sum = 0;
    for (const auto& entry : hist_vec) {
        frequency_sum += entry.second;
    }

    if (frequency_sum != total_count) {
        std::cerr << "\nWARNING: Frequency sum mismatch in " << title << std::endl;
        std::cerr << "  Expected total: " << total_count << std::endl;
        std::cerr << "  Actual sum: " << frequency_sum << std::endl;
        std::cerr << "  Difference: " << (int64_t)frequency_sum - (int64_t)total_count << std::endl;
    }

    std::cout << "\nTotal entries: " << total_count << std::endl;
    std::cout << "Unique values: " << hist_vec.size() << std::endl;
    std::cout << "Frequency sum verification: " << (frequency_sum == total_count ? "PASS" : "FAIL") << std::endl;

    std::cout << "\nDistribution:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(20) << "Category" << std::setw(15) << "Count" << std::setw(12) << "Percentage"
              << "  Bar" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Find max count for scaling
    uint64_t max_count = 0;
    for (const auto& entry : hist_vec) {
        max_count = std::max(max_count, entry.second);
    }

    const int max_bar_width = 30;

    for (const auto& entry : hist_vec) {
        double percentage = (static_cast<double>(entry.second) / total_count) * 100.0;
        int bar_width = static_cast<int>((static_cast<double>(entry.second) / max_count) * max_bar_width);

        std::cout << std::setw(20) << entry.first << std::setw(15) << entry.second;

        // Format percentage: show "< 0.01%" for very small values, "> 99.9%" for very high values
        if (percentage > 0.0 && percentage < 0.01) {
            std::cout << std::setw(11) << "< 0.01%";
        } else if (percentage > 99.9 && percentage < 100.0) {
            std::cout << std::setw(11) << "> 99.9%";
        } else {
            std::cout << std::setw(11) << std::fixed << std::setprecision(2) << percentage << "%";
        }

        std::cout << "  " << std::string(bar_width, '#') << std::endl;
    }

    std::cout << std::string(80, '=') << std::endl;
}

// Generic histogram printer for numeric data
void print_numeric_histogram(const std::map<int, uint64_t>& histogram, uint64_t total_count, const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Convert to vector and sort by value (ascending)
    std::vector<std::pair<int, uint64_t>> hist_vec(histogram.begin(), histogram.end());
    std::sort(hist_vec.begin(), hist_vec.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Verify that frequencies sum to total count
    uint64_t frequency_sum = 0;
    for (const auto& entry : hist_vec) {
        frequency_sum += entry.second;
    }

    if (frequency_sum != total_count) {
        std::cerr << "\nWARNING: Frequency sum mismatch in " << title << std::endl;
        std::cerr << "  Expected total: " << total_count << std::endl;
        std::cerr << "  Actual sum: " << frequency_sum << std::endl;
        std::cerr << "  Difference: " << (int64_t)frequency_sum - (int64_t)total_count << std::endl;
    }

    // Calculate statistics
    int min_val = hist_vec.front().first;
    int max_val = hist_vec.back().first;

    int64_t sum = 0;
    for (const auto& entry : hist_vec) {
        sum += static_cast<int64_t>(entry.first) * entry.second;
    }
    double mean = static_cast<double>(sum) / total_count;

    std::cout << "\nStatistics:" << std::endl;
    std::cout << "Total entries: " << total_count << std::endl;
    std::cout << "Unique values: " << hist_vec.size() << std::endl;
    std::cout << "Frequency sum verification: " << (frequency_sum == total_count ? "PASS" : "FAIL") << std::endl;
    std::cout << "Min: " << min_val << std::endl;
    std::cout << "Max: " << max_val << std::endl;
    std::cout << "Mean: " << std::fixed << std::setprecision(2) << mean << std::endl;

    std::cout << "\nDistribution:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(12) << "Value" << std::setw(15) << "Count" << std::setw(12) << "Percentage"
              << "  Bar" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Find max count for scaling
    uint64_t max_count = 0;
    for (const auto& entry : hist_vec) {
        max_count = std::max(max_count, entry.second);
    }

    const int max_bar_width = 40;

    for (const auto& entry : hist_vec) {
        double percentage = (static_cast<double>(entry.second) / total_count) * 100.0;
        int bar_width = static_cast<int>((static_cast<double>(entry.second) / max_count) * max_bar_width);

        std::cout << std::setw(12) << entry.first << std::setw(15) << entry.second;

        // Format percentage: show "< 0.01%" for very small values, "> 99.9%" for very high values
        if (percentage > 0.0 && percentage < 0.01) {
            std::cout << std::setw(11) << "< 0.01%";
        } else if (percentage > 99.9 && percentage < 100.0) {
            std::cout << std::setw(11) << "> 99.9%";
        } else {
            std::cout << std::setw(11) << std::fixed << std::setprecision(2) << percentage << "%";
        }

        std::cout << "  " << std::string(bar_width, '#') << std::endl;
    }

    std::cout << std::string(80, '=') << std::endl;
}

void print_ulp_histogram(const std::map<uint64_t, uint64_t>& ulp_histogram, uint64_t total_mismatches) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ULP DIFFERENCE HISTOGRAM" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Convert to vector and sort by ULP value
    std::vector<std::pair<uint64_t, uint64_t>> ulp_vec(ulp_histogram.begin(), ulp_histogram.end());
    std::sort(ulp_vec.begin(), ulp_vec.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;  // Sort by ULP value ascending
    });

    // Verify that frequencies sum to total count
    uint64_t frequency_sum = 0;
    for (const auto& entry : ulp_vec) {
        frequency_sum += entry.second;
    }

    if (frequency_sum != total_mismatches) {
        std::cerr << "\nWARNING: Frequency sum mismatch in ULP DIFFERENCE HISTOGRAM" << std::endl;
        std::cerr << "  Expected total: " << total_mismatches << std::endl;
        std::cerr << "  Actual sum: " << frequency_sum << std::endl;
        std::cerr << "  Difference: " << (int64_t)frequency_sum - (int64_t)total_mismatches << std::endl;
    }

    std::cout << "\nStatistics:" << std::endl;
    std::cout << "Total mismatches: " << total_mismatches << std::endl;
    std::cout << "Unique ULP differences: " << ulp_vec.size() << std::endl;
    std::cout << "Frequency sum verification: " << (frequency_sum == total_mismatches ? "PASS" : "FAIL") << std::endl;

    // Find min, max, and calculate mean
    uint64_t min_ulp = ulp_vec.front().first;
    uint64_t max_ulp = ulp_vec.back().first;

    uint64_t sum_ulp = 0;
    for (const auto& entry : ulp_vec) {
        sum_ulp += entry.first * entry.second;
    }
    double mean_ulp = static_cast<double>(sum_ulp) / total_mismatches;

    std::cout << "Min ULP: " << min_ulp << std::endl;
    std::cout << "Max ULP: " << max_ulp << std::endl;
    std::cout << "Mean ULP: " << std::fixed << std::setprecision(2) << mean_ulp << std::endl;

    // Print histogram
    std::cout << "\nHistogram:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(12) << "ULP Diff" << std::setw(15) << "Count" << std::setw(12) << "Percentage"
              << "  Bar" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Find max count for scaling the bar chart
    uint64_t max_count = 0;
    for (const auto& entry : ulp_vec) {
        max_count = std::max(max_count, entry.second);
    }

    const int max_bar_width = 40;

    for (const auto& entry : ulp_vec) {
        uint64_t ulp_diff = entry.first;
        uint64_t count = entry.second;
        double percentage = (static_cast<double>(count) / total_mismatches) * 100.0;

        // Calculate bar width
        int bar_width = static_cast<int>((static_cast<double>(count) / max_count) * max_bar_width);

        std::cout << std::setw(12) << ulp_diff << std::setw(15) << count;

        // Format percentage: show "< 0.01%" for very small values, "> 99.9%" for very high values
        if (percentage > 0.0 && percentage < 0.01) {
            std::cout << std::setw(11) << "< 0.01%";
        } else if (percentage > 99.9 && percentage < 100.0) {
            std::cout << std::setw(11) << "> 99.9%";
        } else {
            std::cout << std::setw(11) << std::fixed << std::setprecision(2) << percentage << "%";
        }

        std::cout << "  " << std::string(bar_width, '#') << std::endl;
    }

    std::cout << std::string(80, '=') << std::endl;

    // Print cumulative statistics for common ULP ranges
    std::cout << "\nCumulative Distribution:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<uint64_t> thresholds = {1, 2, 5, 10, 50, 100, 1000, 10000};
    uint64_t cumulative = 0;

    for (uint64_t threshold : thresholds) {
        uint64_t count_up_to = 0;
        for (const auto& entry : ulp_vec) {
            if (entry.first <= threshold) {
                count_up_to += entry.second;
            }
        }
        if (count_up_to > cumulative) {
            cumulative = count_up_to;
            double percentage = (static_cast<double>(cumulative) / total_mismatches) * 100.0;
            std::cout << "ULP <= " << std::setw(6) << threshold << ": " << std::setw(10) << cumulative << " ("
                      << std::fixed << std::setprecision(2) << std::setw(6) << percentage << "%)" << std::endl;
        }
        if (cumulative == total_mismatches) {
            break;
        }
    }

    // Print remaining if not all covered
    if (cumulative < total_mismatches) {
        std::cout << "ULP >  " << std::setw(6) << thresholds.back() << ": " << std::setw(10)
                  << (total_mismatches - cumulative) << " (" << std::fixed << std::setprecision(2) << std::setw(6)
                  << ((static_cast<double>(total_mismatches - cumulative) / total_mismatches) * 100.0) << "%)"
                  << std::endl;
    }

    std::cout << std::string(50, '-') << std::endl;
}

static inline double fpu_float_to_double(uint32_t u) {
    uint64_t s = uint64_t(u & 0x80000000) << 32;
    uint64_t em = u & 0x7FFFFFFF;
    if (em < 0x800000) {
        return std::bit_cast<double>(s);
    }
    em += uint64_t(1023 - 127) << 23;
    return std::bit_cast<double>(s | (em << 29));
}

static inline uint32_t double_to_fpu_float(double d) { return std::bit_cast<uint32_t>(float(d)); }

int main(int argc, char* argv[]) {
    // Default filename
    std::string filename;  // = "ttnn_add_mismatches.bin";

    // Allow filename to be passed as command-line argument
    if (argc > 1) {
        filename = argv[1];
    }

    // Open the binary file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return 1;
    }

    // Read the 8-byte header (number of mismatches)
    uint64_t num_mismatches;
    file.read(reinterpret_cast<char*>(&num_mismatches), sizeof(uint64_t));

    if (!file) {
        std::cerr << "Error: Could not read header from file" << std::endl;
        return 1;
    }

    std::cout << "Number of mismatches: " << num_mismatches << std::endl;
    std::cout << "File: " << filename << std::endl;
    std::cout << "Expected file size: " << (8 + num_mismatches * 4 * 2) << " bytes" << std::endl;
    std::cout << std::endl;

    // Map to count frequency of each input_a value (left operand only)
    std::map<uint16_t, uint64_t> frequency_map_a;

    // Map to count frequency across both operands (combined)
    std::map<uint16_t, uint64_t> frequency_map_combined;

    // Map to store ULP difference histogram
    std::map<uint64_t, uint64_t> ulp_histogram;

    // Map to store all mismatch records grouped by left operand (input_a)
    std::map<uint16_t, std::vector<MismatchRecord>> records_by_operand_a;

    // Vector to store ALL mismatch records (for filtered CSV generation)
    std::vector<MismatchRecord> all_records;

    // Additional histograms for analysis
    std::map<std::string, uint64_t> signs_same_histogram;
    std::map<int, uint64_t> exp_diff_histogram;
    std::map<int, uint64_t> larger_exp_histogram;
    std::map<int, uint64_t> smaller_exp_histogram;
    std::map<std::string, uint64_t> left_type_histogram;
    std::map<std::string, uint64_t> right_type_histogram;
    std::map<int, uint64_t> left_exp_histogram;
    std::map<int, uint64_t> right_exp_histogram;

    std::map<int, uint64_t> diff_9_a_exp_histogram;
    std::map<int, uint64_t> diff_9_result_is_smaller;

    // Read all mismatch records
    for (uint64_t i = 0; i < num_mismatches; ++i) {
        uint16_t input_a, input_b, result, expected;

        // Read the 4 uint16 values for this mismatch record
        file.read(reinterpret_cast<char*>(&input_a), sizeof(uint16_t));
        file.read(reinterpret_cast<char*>(&input_b), sizeof(uint16_t));
        file.read(reinterpret_cast<char*>(&result), sizeof(uint16_t));
        file.read(reinterpret_cast<char*>(&expected), sizeof(uint16_t));

        if (!file) {
            std::cerr << "Error: Could not read mismatch record " << i << std::endl;
            return 1;
        }
        /*
                double a_d = fpu_float_to_double(input_a << 16);
                double b_d = fpu_float_to_double(input_b << 16);
                double c_d = a_d + b_d;
                uint32_t c_f = double_to_fpu_float(c_d);
                uint16_t dst = (c_f + 0x8000) >> 16;
                printf("input_a=0x%04x input_b=0x%04x result=0x%04x expected=0x%04x\n", input_a, input_b, result,
           expected); printf("\tCPU a=" "%" PRIx64 " b=" "%" PRIx64 " c=" "%" PRIx64 " c_f=" "%" PRIx32 " dst=0x%04x"
           "\n", std::bit_cast<uint64_t>(a_d), std::bit_cast<uint64_t>(b_d), std::bit_cast<uint64_t>(c_d), c_f, dst);
        */
        // Count the frequency of the left operand (input_a) only
        frequency_map_a[input_a]++;

        // Count the frequency across both operands
        frequency_map_combined[input_a]++;
        frequency_map_combined[input_b]++;

        // Calculate and store ULP difference
        uint64_t ulp_diff = calculate_ulp_difference(result, expected);
        ulp_histogram[ulp_diff]++;

        // Store the mismatch record
        MismatchRecord record = {input_a, input_b, result, expected};
        records_by_operand_a[input_a].push_back(record);
        all_records.push_back(record);

        // Collect additional histogram data
        BFloat16Info left_info(input_a);
        BFloat16Info right_info(input_b);

        // Signs same histogram
        bool signs_same = (left_info.sign == right_info.sign);
        signs_same_histogram[signs_same ? "Yes" : "No"]++;

        // Exponent histograms
        uint8_t larger_exp = std::max(left_info.exponent, right_info.exponent);
        uint8_t smaller_exp = std::min(left_info.exponent, right_info.exponent);
        int exp_diff = static_cast<int>(larger_exp) - static_cast<int>(smaller_exp);

        exp_diff_histogram[exp_diff]++;
        larger_exp_histogram[static_cast<int>(larger_exp)]++;
        smaller_exp_histogram[static_cast<int>(smaller_exp)]++;

        if (exp_diff == 9) {
            diff_9_a_exp_histogram[static_cast<int>(left_info.exponent)]++;
            BFloat16Info result_info(result);
            BFloat16Info expected_info(expected);
            if (result_info.mantissa < expected_info.mantissa) {
                diff_9_result_is_smaller[0]++;
            }
            if (result_info.mantissa > expected_info.mantissa) {
                diff_9_result_is_smaller[1]++;
            }
            if (result_info.mantissa == expected_info.mantissa) {
                diff_9_result_is_smaller[2]++;
            }
        }

        // Individual operand exponent histograms
        left_exp_histogram[static_cast<int>(left_info.exponent)]++;
        right_exp_histogram[static_cast<int>(right_info.exponent)]++;

        // Type histograms
        left_type_histogram[left_info.get_type()]++;
        right_type_histogram[right_info.get_type()]++;
    }

    file.close();

    std::cout << "Total unique input_a values with mismatches: " << frequency_map_a.size() << std::endl;
    std::cout << "Total unique values (both operands) with mismatches: " << frequency_map_combined.size() << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // TABLE 1: Top 100 most frequent input_a (left operand) values
    // ========================================================================

    // Convert map to vector for sorting
    std::vector<std::pair<uint16_t, uint64_t>> frequency_vec_a(frequency_map_a.begin(), frequency_map_a.end());

    // Sort by frequency (descending)
    std::sort(frequency_vec_a.begin(), frequency_vec_a.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Display the top 100 (or fewer if less than 100 unique values)
    int num_to_display_a = std::min(100, static_cast<int>(frequency_vec_a.size()));
    print_frequency_table(
        frequency_vec_a,
        num_mismatches,
        "Top " + std::to_string(num_to_display_a) + " most frequent input_a (left operand) values:",
        num_to_display_a);

    // ========================================================================
    // TABLE 2: Top 100 most frequent values across both operands
    // ========================================================================

    // Convert map to vector for sorting
    std::vector<std::pair<uint16_t, uint64_t>> frequency_vec_combined(
        frequency_map_combined.begin(), frequency_map_combined.end());

    // Sort by frequency (descending)
    std::sort(frequency_vec_combined.begin(), frequency_vec_combined.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Display the top 100 (or fewer if less than 100 unique values)
    int num_to_display_combined = std::min(100, static_cast<int>(frequency_vec_combined.size()));

    // Note: total count for combined is 2 * num_mismatches since each mismatch has 2 operands
    uint64_t total_combined_count = 2 * num_mismatches;
    print_frequency_table(
        frequency_vec_combined,
        total_combined_count,
        "Top " + std::to_string(num_to_display_combined) + " most frequent values (both operands combined):",
        num_to_display_combined);

    // ========================================================================
    // ULP DIFFERENCE HISTOGRAM
    // ========================================================================

    print_ulp_histogram(ulp_histogram, num_mismatches);

    // ========================================================================
    // ADDITIONAL HISTOGRAMS
    // ========================================================================

    print_categorical_histogram(signs_same_histogram, num_mismatches, "SIGNS SAME HISTOGRAM");
    print_numeric_histogram(exp_diff_histogram, num_mismatches, "EXPONENT DIFFERENCE HISTOGRAM");
    print_numeric_histogram(larger_exp_histogram, num_mismatches, "LARGER EXPONENT HISTOGRAM");
    print_numeric_histogram(smaller_exp_histogram, num_mismatches, "SMALLER EXPONENT HISTOGRAM");
    print_numeric_histogram(left_exp_histogram, num_mismatches, "LEFT OPERAND EXPONENT HISTOGRAM");
    print_numeric_histogram(right_exp_histogram, num_mismatches, "RIGHT OPERAND EXPONENT HISTOGRAM");
    print_numeric_histogram(diff_9_a_exp_histogram, num_mismatches, "EXP DIFF 9 EXPONENT HISTOGRAM");
    print_numeric_histogram(diff_9_result_is_smaller, num_mismatches, "EXP DIFF Resul Smaller HISTOGRAM");
    print_categorical_histogram(left_type_histogram, num_mismatches, "LEFT OPERAND TYPE HISTOGRAM");
    print_categorical_histogram(right_type_histogram, num_mismatches, "RIGHT OPERAND TYPE HISTOGRAM");

    // ========================================================================
    // GENERATE CSV FILES FOR TOP 10 MOST FREQUENT VALUES
    // ========================================================================

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "GENERATING CSV FILES FOR TOP 10 MOST FREQUENT VALUES" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Create output directory
    std::string output_dir = "mismatch_csvs";
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    int mkdir_result = system(mkdir_cmd.c_str());
    (void)mkdir_result;  // Suppress unused variable warning

    // Get top 10 most frequent values from the combined frequency map
    // (frequency_vec_combined is already sorted by frequency descending from earlier)
    int num_csv_to_generate = std::min(10, static_cast<int>(frequency_vec_combined.size()));

    std::cout << "\nGenerating CSV files for top " << num_csv_to_generate
              << " most frequent values (either operand):" << std::endl;

    // Build a map to store all records for each value (both as left and right operand)
    std::map<uint16_t, std::vector<MismatchRecord>> records_by_value;

    // Collect records where this value appears (either as left or right operand)
    for (const auto& operand_entry : records_by_operand_a) {
        for (const auto& record : operand_entry.second) {
            // Add to left operand value
            records_by_value[record.input_a].push_back(record);
            // Add to right operand value (avoiding duplicates if both are same)
            if (record.input_a != record.input_b) {
                records_by_value[record.input_b].push_back(record);
            }
        }
    }

    // Generate CSV files for top 10 values
    for (int i = 0; i < num_csv_to_generate; ++i) {
        uint16_t value = frequency_vec_combined[i].first;
        uint64_t frequency = frequency_vec_combined[i].second;

        std::cout << "  " << (i + 1) << ". Value 0x" << std::hex << std::setfill('0') << std::setw(4) << value
                  << std::dec << " (frequency: " << frequency << ")";

        if (records_by_value.find(value) != records_by_value.end()) {
            generate_csv_for_operand(value, records_by_value[value], output_dir);
            std::cout << " - CSV generated" << std::endl;
        } else {
            std::cout << " - No records found (warning!)" << std::endl;
        }
    }

    std::cout << "\nTotal CSV files generated: " << num_csv_to_generate << std::endl;
    std::cout << "Location: ./" << output_dir << "/" << std::endl;
    std::cout << "Filename format: operand_a_0xXXXX.csv" << std::endl;
    std::cout << "Note: Each CSV contains all mismatches where that value appears (as either left or right operand)"
              << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // ========================================================================
    // GENERATE FILTERED CSV FILE
    // ========================================================================

    generate_filtered_csv(all_records, output_dir, 100);

    return 0;
}
