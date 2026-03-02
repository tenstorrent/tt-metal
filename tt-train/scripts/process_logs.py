import csv

from pydantic import BaseModel, ConfigDict


class RunMetadata(BaseModel):
    test_ts: int
    model_name: str
    model_filename: str
    binary_name: str
    args: str
    git_commit_hash: str

    model_config = ConfigDict(protected_namespaces=())


class MemoryTracking(BaseModel):
    metadata: RunMetadata
    model_dram_mb: float
    optimizer_dram_mb: float
    activations_dram_mb: float
    gradients_dram_mb: float
    unaccounted_dram_mb: float
    total_dram_mb: float
    device_memory_mb: float

    model_config = ConfigDict(protected_namespaces=())


def normalize_json(data: dict) -> dict:
    new_data = dict()
    for key, value in data.items():
        if not isinstance(value, dict):
            new_data[key] = value
        else:
            for k, v in value.items():
                # new_data[key + "_" + k] = v
                # k assume it's unique in nested
                new_data[k] = v
    return new_data


def write_csv(pydantic_model, log_filename):
    pydantic_json = pydantic_model.model_dump(mode="json")
    normalized_json = normalize_json(pydantic_json)

    with open(f"{log_filename}.csv", "w", newline="") as csvfile:
        fieldnames = normalized_json.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows([normalized_json])
        print(f"\nWritten {log_filename}.csv")
