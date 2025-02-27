#include "device_worker.hpp"

DeviceWorker::DeviceWorker(IDevice* device) : device_(device), stop_(false), tasks_(8192) {
    worker_thread_ = std::thread(&DeviceWorker::process_tasks, this);
}

DeviceWorker::~DeviceWorker() {
    stop_ = true;
    worker_thread_.join();
}

void DeviceWorker::enqueue_task(std::function<void()> task) { tasks_.push(new std::function<void()>(std::move(task))); }

void DeviceWorker::process_tasks() {
    while (!stop_) {
        std::function<void()>* task_ptr;
        if (tasks_.pop(task_ptr)) {
            std::unique_ptr<std::function<void()>> task(task_ptr);
            (*task)();
        } else {
            std::this_thread::yield();
        }
    }
}
