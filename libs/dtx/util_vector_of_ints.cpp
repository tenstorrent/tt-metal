#include "util_vector_of_ints.hpp"


// ========================================================
//                      PRINT / DEBUG
// ========================================================

string v2s(vector<int> vector) {
    string output_string;
    output_string.append("[");
    for (int i=0; i<vector.size(); i++) {
        output_string.append(to_string(vector[i]));
        if (i < vector.size()-1) output_string.append(",");
    }
    output_string.append("]");
    return output_string;
}

string s(int num_spaces) {
    string out;
    for (int i=0; i<num_spaces; i++) {
        out.append(" ");
    }
    return out;
}
string sp(int num_spaces) {
    string out;
    for (int i=0; i<num_spaces; i++) {
        out.append(" ");
    }
    return out;
}

// ========================================================
//                      VECTOR OF INTS
// ========================================================

vector<int> vector_addition(vector<int> a, vector<int> b, int const_value) {
    vector<int> result;
    for (int d=0; d<a.size(); d++) {
        result.push_back(a[d] + b[d] + const_value);
    }
    return result;
 }

vector<int> vector_subtraction(vector<int> a, vector<int> b) {
    vector<int> result;
    for (int d=0; d<a.size(); d++) {
        result.push_back(a[d] - b[d]);
    }
    return result;
}

vector<int> vector_division(vector<int> a, vector<int> b) {
    vector<int> result;
    for (int d=0; d<a.size(); d++) {
        assert(a[d] % b[d] == 0);
        result.push_back(a[d] / b[d]);
    }
    return result;
}

vector<int> vector_multiplication(vector<int> a, vector<int> b) {
    vector<int> result;
    for (int d=0; d<a.size(); d++) {
        result.push_back(a[d] * b[d]);
    }
    return result;
}

int vector_product(vector<int> a) {
    if (a.size() == 0) return 0;
    int result = 1;
    for (int d=0; d<a.size(); d++) {
        result = result * a[d];
    }
    return result;
}

vector<int> copy_vector_of_ints(vector<int> a) {
    vector<int> result(a);
    return result;
}

vector<int> copy(vector<int> a) {
    vector<int> result(a);
    return result;
}

vector<int> zeros(int rank){
    vector<int> zeros;
    for (int d=0; d<rank; d++){
        zeros.push_back(0);
    }
    return zeros;
}

vector<int> ones(int rank){
    vector<int> zeros;
    for (int d=0; d<rank; d++){
        zeros.push_back(1);
    }
    return zeros;
}

vector<int> vector_pad_on_left(vector<int> input, int pad_size, int pad_value){
    vector<int> output;
    for (int d=0; d<pad_size; d++){
        output.push_back(pad_value);
    }
    for (int d=0; d<input.size(); d++){
        output.push_back(input[d]);
    }
    return output;
}

vector<int> increment(vector<int> current, vector<int> start, vector<int> end) {
    bool DEBUG = false;

    if (DEBUG) {
        cout << "\n\nIncrement" << endl;
        cout << "range: " << v2s(start) << " --> " << v2s(end) << endl;
        cout << "current: " << v2s(current) << endl;
    }

    vector<int> next = current;
    int rank = next.size();
    next[rank-1]++;

    for (int d=rank-1; d>   0; d--) {
        if (DEBUG) cout << d << endl;

        if (next[d] > end[d]) {
           next[d] = start[d]; // resent current
           next[d-1]++;
        }
    }

    if (DEBUG) cout << "next: " << v2s(next) << endl;
    return next;
}
