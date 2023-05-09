#include <mutex>
#include <thread>
#include <queue>

template <class T>
class TSQueue {
    public:
        TSQueue();
        TSQueue(uint32_t capacity);

        void push(T e);
        T peek();
        void pop();

        size_t size();
        std::queue<T> q;
        std::condition_variable empty_condition;
        std::condition_variable full_condition;
        uint32_t capacity;

    private:
        std::mutex m;
};


template <class T>
TSQueue<T>::TSQueue(uint32_t capacity) {
    this->q = std::queue<T>();
    this->capacity = capacity;
}

template <class T>
TSQueue<T>::TSQueue() {
    this->q = std::queue<T>();
    this->capacity = 100;
}

template <class T>
void TSQueue<T>::push(T t) {

    std::unique_lock<std::mutex> lock(this->m);

    this->full_condition.wait(lock, [this]() { return this->q.size() < this->capacity; });

    this->q.push(t);

    this->empty_condition.notify_one();
}

template <class T>
T TSQueue<T>::peek() {
    std::unique_lock<std::mutex> lock(this->m);

    this->empty_condition.wait(lock, [this]() { return !this->q.empty(); });

    T t = this->q.front();

    this->full_condition.notify_one();
    return t;
}

template <class T>
void TSQueue<T>::pop() {
    std::unique_lock<std::mutex> lock(this->m);

    this->empty_condition.wait(lock, [this]() { return !this->q.empty(); });

    this->q.pop();

    this->full_condition.notify_one();
}

template <class T>
size_t TSQueue<T>::size() { return this->q.size(); }
