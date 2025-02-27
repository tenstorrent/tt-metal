#pragma once

#include <thread>
#include <boost/lockfree/queue.hpp>
#include <atomic>
#include <functional>
#include <memory>
#include <tt-metalium/mesh_device.hpp>

class DeviceWorker {
public:
    DeviceWorker(IDevice* device);
    ~DeviceWorker();
    void enqueue_task(std::function<void()> task);

private:
    void process_tasks();

    IDevice* device_;
    std::thread worker_thread_;
    boost::lockfree::queue<std::function<void()>*, boost::lockfree::fixed_sized<true>> tasks_;
    std::atomic<bool> stop_;
};
