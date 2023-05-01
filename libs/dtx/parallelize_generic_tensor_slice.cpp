#include "dtx.hpp"
#include "util_vector_of_ints.hpp"

vector<vector<vector<int>>> generate_sliced_ranges(vector<int> shape, vector<int> slice_factors){
    bool DEBUG = true;


    if (shape.size() != slice_factors.size()) throw std::runtime_error("shape and slice_factor dont have the same rank!");
    int rank = shape.size();

    if (DEBUG) cout << "\n\nPart 3 - Count the ND slices:" << endl;
    if (DEBUG) cout << s(2) << "shape=" << v2s(shape) << ", slice_factors=" << v2s(slice_factors) << endl;

    vector<vector<vector<int>>> list_of_subtensor_ranges;

    vector<int> slice_size;
    for (int d=0; d<rank; d++){
            slice_size.push_back(shape[d] / slice_factors[d]);
    }
    if (DEBUG) cout << s(2) << "slice_size = " << v2s(slice_size) << endl;

    int volume = vector_product(slice_factors);
    if (DEBUG) cout << s(2) << "volume = " << volume << endl;

    vector<int> counter = zeros(rank);
    for (int i=0; i<volume; i++){

        vector<int> start;
        vector<int> end;
        for (int d=0; d<rank; d++) {
            start.push_back(counter[d] * slice_size[d]);
            end.push_back(counter[d] * slice_size[d] + slice_size[d] - 1);
        }
        if (DEBUG) cout << s(2) << "i = " << i << ".  counter = " << v2s(counter) << ";  " << v2s(start) << " => " << v2s(end) << endl;

        counter.back()++;
        for (int d=rank-1; d>0; d--) {
            if (counter[d] == slice_factors[d]) {
                counter[d-1]++;
                counter[d] = 0;
            }
        }
        list_of_subtensor_ranges.push_back({start, end});
    }
    return list_of_subtensor_ranges;
}

vector<vector<int>> generate_list_of_cores_based_on_range(vector<int> cores_start, vector<int> cores_end){
    bool DEBUG = false;
    if (DEBUG) cout << "\nGenerating list of cores:  " << v2s(cores_start) << " --> " << v2s(cores_end) << endl;

    vector<vector<int>> list_of_cores;
    vector<int> cores_shape;
    for (int d=0; d<cores_start.size(); d++){
        cores_shape.push_back(cores_end[d] - cores_start[d] +1);
    }
    int number_of_cores = vector_product(cores_shape);
    if (DEBUG) cout << s(2) << "cores shape =" << v2s(cores_shape) << endl;
    if (DEBUG) cout << s(2) << "number of cores =" << number_of_cores << endl;
    if (DEBUG) cout << s(2) << "core list:" << number_of_cores << endl;
    vector<int> core = cores_start;
    for (int i=0; i<number_of_cores; i++){
        list_of_cores.push_back(core);
        if (DEBUG) cout << s(4) << i << ". " << v2s(core) << endl;
        core[1]++;
        if (core[1] > cores_end[1]) {
            core[0]++;
            core[1] = cores_start[1];
        }
    }
    return list_of_cores;
}


bool parallelize_generic_tensor_slice(DataTransformations * dtx, vector<int> slice_factors, vector<int> cores_start, vector<int> cores_end) {
    bool DEBUG = true;

    // Set up producer node
    TransformationNode * producer = dtx->transformations.back();
    if (producer->groups.size() > 1) throw std::runtime_error("The paralleliztion slicing transformation can be added only if the producer transformation has 1 group of TensorPairs.");
    vector<int> shape = producer->groups[0]->shape;
    int rank = shape.size();
    vector<int> slice_shape = vector_division(shape, slice_factors);

    // Create consumer node
    int number_of_groups = vector_product(slice_factors);
    TransformationNode * consumer = new TransformationNode("parallelization", number_of_groups);
    dtx->transformations.push_back(consumer);

    vector<vector<vector<int>>> list_of_subtensor_ranges = generate_sliced_ranges(shape, slice_factors);
    vector<vector<int>> list_of_cores = generate_list_of_cores_based_on_range(cores_start, cores_end);

    // Populate the new TransformationNode with the slices
    for (int group_idx=0; group_idx<number_of_groups; group_idx++){
        //                                                                                          SRC                                         DST
        consumer->groups[group_idx]->tensor_pairs.push_back(new TensorPair(new DTXTensor({list_of_subtensor_ranges[group_idx][0]},  {list_of_subtensor_ranges[group_idx][1]}),
                                                                           0,
                                                                           new DTXTensor(zeros(rank),  slice_shape)));
        consumer->groups[group_idx]->core = list_of_cores[group_idx];
        consumer->groups[group_idx]->shape = slice_shape;
    }

    if (DEBUG) dtx->print();

    return true;
}
